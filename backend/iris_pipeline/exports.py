import csv
import json
import os
import threading
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from .media import IMAGES, read_image, write_png


class Exporter:
    def __init__(self, settings, db, manager):
        self.settings, self.db, self.manager = settings, db, manager
        self.lock = threading.Lock()

    def export(self, job_id, kind):
        with self.lock:
            job = self.db.job(job_id)
            if job["status"] in {"running", "queued", "cancelling"}:
                raise ValueError("Wait for processing to finish before exporting")
            extension = kind if kind in {"csv", "json"} else "zip"
            target = self.manager.path(job_id, f"exports/{kind}.{extension}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if kind == "json":
                with target.open("w", encoding="utf-8") as handle:
                    handle.write('{"job":' + json.dumps(job, ensure_ascii=False) + ',"runs":' + json.dumps(self.db.runs(job_id), ensure_ascii=False) + ',"detections":[')
                    first = True
                    for item in self.db.iter_detections(job_id):
                        handle.write(("" if first else ",") + json.dumps({**item, "history": self.db.audit(item["id"])}, ensure_ascii=False))
                        first = False
                    handle.write("]}")
            elif kind == "csv":
                columns = ["id", "source_id", "source_name", "frame_index", "timestamp", "bbox", "raw_text", "corrected_text",
                           "digits", "arabic_letters", "detector_confidence", "ocr_confidence", "needs_review", "review_flags",
                           "width", "height", "brightness", "contrast", "char_sharpness", "edge_strength", "edge_density", "quality_score"]
                with target.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    for item in self.db.iter_detections(job_id):
                        writer.writerow({**item, **item["quality"], "bbox": json.dumps(item["bbox"]), "review_flags": "; ".join(item["review_flags"])})
            elif kind == "crops":
                manifest = self.manager.path(job_id, "exports/manifest.jsonl")
                with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive, manifest.open("w", encoding="utf-8") as handle:
                    for item in self.db.iter_detections(job_id):
                        for relative in [item["original"], *[h["path"] for h in self.db.audit(item["id"])["enhancements"]]]:
                            archive.write(self.manager.path(job_id, relative), relative)
                        handle.write(json.dumps({**item, "history": self.db.audit(item["id"])}, ensure_ascii=False)+"\n")
                    handle.flush()
                    archive.write(manifest, "manifest.jsonl")
                    archive.writestr("job.json", json.dumps({"job": job, "runs": self.db.runs(job_id)}, ensure_ascii=False))
            elif kind == "annotated":
                self.annotated(job, target)
            else:
                raise ValueError("Unsupported export format")
            return target

    def font(self):
        paths = [self.settings.annotation_font, str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"),
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for path in paths:
            if path and Path(path).is_file():
                layout = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
                return ImageFont.truetype(path, 22, layout_engine=layout)
        raise ValueError("Set IRIS_ANNOTATION_FONT to an Arabic-capable TrueType font")

    def draw(self, image, detections, font):
        if not detections:
            return image
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for item in detections:
            x1, y1, x2, y2 = item["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline="#62d98f", width=3)
            text = item.get("corrected_text")
            if text is None:
                text = item.get("digits", "") + " " + item.get("arabic_letters", "")
            if not text.strip():
                text = "Plate"
            kwargs = {"direction": "rtl"} if features.check("raqm") else {}
            if not kwargs:
                import arabic_reshaper
                from bidi.algorithm import get_display
                text = get_display(arabic_reshaper.reshape(text))
            bbox = draw.textbbox((0, 0), text, font=font, **kwargs)
            top = max(0, y1-(bbox[3]-bbox[1])-12)
            draw.rectangle((x1, top, min(pil.width, x1+bbox[2]-bbox[0]+12), y1), fill="#123425")
            draw.text((x1+6, top+3-bbox[1]), text, font=font, fill="white", **kwargs)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def annotated(self, job, target):
        font = self.font()
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
            for source in job["sources"]:
                source_path = self.manager.path(job["id"], source["stored"])
                if source_path.suffix.lower() in IMAGES:
                    rows = self.db.detections(job["id"], limit=100000, source=source["id"])["items"]
                    output = self.manager.path(job["id"], f"exports/{source['id']}.png")
                    write_png(output, self.draw(read_image(source_path), rows, font))
                else:
                    output = self.annotated_video(job, source, source_path, font)
                archive.write(output, output.name)
            archive.writestr("sources.json", json.dumps(job["sources"], ensure_ascii=False))

    def annotated_video(self, job, source, source_path, font):
        import av
        from fractions import Fraction
        output = self.manager.path(job["id"], f"exports/{source['id']}.mp4")
        # Source frame timestamps survive re-encoding, including variable frame rates.
        with av.open(str(source_path)) as incoming, av.open(str(output), "w") as outgoing:
            video = incoming.streams.video[0]
            stream = outgoing.add_stream("libx264", rate=video.average_rate or Fraction(30))
            stream.width, stream.height = video.width, video.height
            stream.pix_fmt = "yuv420p" if video.width % 2 == 0 and video.height % 2 == 0 else "yuv444p"
            stream.time_base = video.time_base
            stream.codec_context.time_base = video.time_base
            stream.options = {"crf": "20", "preset": "fast"}
            audio_map = {s.index: outgoing.add_stream_from_template(s) for s in incoming.streams.audio}
            with self.db.connect() as db:
                cursor = db.execute("SELECT payload FROM detections WHERE job_id=? AND source_id=? ORDER BY rowid", (job["id"], source["id"]))
                def groups():
                    group, current = [], None
                    for row in cursor:
                        item = json.loads(row[0])
                        if current is not None and item["frame_index"] != current:
                            yield current, group
                            group = []
                        current = item["frame_index"]
                        group.append(item)
                    if group:
                        yield current, group
                iterator = iter(groups())
                next_group = next(iterator, None)
                active, active_time, index = [], -100, 0
                origin = None
                for packet in incoming.demux():
                    if packet.stream.type == "audio" and packet.dts is not None:
                        packet.stream = audio_map[packet.stream.index]
                        outgoing.mux(packet)
                    elif packet.stream.index == video.index:
                        for frame in packet.decode():
                            if origin is None:
                                origin = float(frame.time or 0)
                            timestamp = float(frame.time or 0)-origin
                            if next_group and index == next_group[0]:
                                active, active_time = next_group[1], timestamp
                                next_group = next(iterator, None)
                            if timestamp-active_time >= 1/job.get("input_options", job["options"])["sample_fps"]-1e-6:
                                active = []
                            rendered = av.VideoFrame.from_ndarray(self.draw(frame.to_ndarray(format="bgr24"), active, font), format="bgr24")
                            rendered.pts, rendered.time_base = frame.pts, frame.time_base
                            for encoded in stream.encode(rendered):
                                outgoing.mux(encoded)
                            index += 1
                for encoded in stream.encode():
                    outgoing.mux(encoded)
        return output
