"""Run real models through the live API and save evidence in .runtime/."""
import argparse
import io
import json
import time
import zipfile
from fractions import Fraction
from pathlib import Path
import av
import cv2
import httpx
import numpy as np
from iris_pipeline.config import ROOT
from iris_pipeline.media import read_image, write_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=Path, required=True, help='Folder containing labeled plate-crop subfolders, named digits logical-Arabic')
    parser.add_argument('--url', default='http://127.0.0.1:8014')
    args = parser.parse_args()
    target = ROOT/'.runtime'/'verification-media'
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image in sorted(args.samples.rglob('*')):
        if image.suffix.lower() not in {'.png','.jpg','.jpeg'}:
            continue
        label = image.parent.name.split(' ',1)
        if len(label) != 2 or not label[0].isdigit():
            continue
        copied = target/f'plate-{len(manifest):03d}.png'
        write_png(copied, read_image(image))
        manifest.append({'path':copied.name, 'truth':label[0]+label[1][::-1]})
        if len(manifest) >= 6:
            break
    if not manifest:
        raise ValueError('No labeled samples found')
    labels = target/'labels.json'
    labels.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    canvas = np.full((360,640,3),90,dtype=np.uint8)
    plate = read_image(target/manifest[0]['path'])
    height = 85
    width = min(300,round(plate.shape[1]*height/plate.shape[0]))
    plate = cv2.resize(plate,(width,height))
    canvas[220:220+height,70:70+width] = plate
    canvas[90:90+height,330:330+width] = plate
    full_image = target/'two-plates.png'
    write_png(full_image,canvas)
    video_path = target/'two-plates.mp4'
    with av.open(str(video_path),'w') as container:
        stream = container.add_stream('libx264',rate=30)
        stream.width,stream.height,stream.pix_fmt = 640,360,'yuv420p'
        for index in range(60):
            frame = av.VideoFrame.from_ndarray(canvas,format='bgr24')
            frame.pts, frame.time_base = index,Fraction(1,30)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    evidence = {}
    with httpx.Client(base_url=args.url,timeout=180) as client:
        def call(method,path,**kwargs):
            response = client.request(method,path,**kwargs)
            response.raise_for_status()
            return response.json()
        def wait(job):
            deadline = time.monotonic()+240
            while time.monotonic()<deadline:
                result = call('GET',f'/api/jobs/{job["id"]}')
                if result['status'] in {'completed','failed','cancelled'}:
                    assert result['status']=='completed',result.get('error')
                    return result
                time.sleep(.2)
            raise TimeoutError('Real inference job did not complete')
        deadline = time.monotonic()+120
        evidence['health'] = {}
        while time.monotonic()<deadline:
            try:
                evidence['health'] = call('GET','/api/health')
                if evidence['health'].get('state') == 'ready':
                    break
            except httpx.RequestError:
                pass
            time.sleep(1)
        assert evidence['health'].get('state')=='ready',evidence['health']
        assert evidence['health']['gpu']['probe']==32
        assert evidence['health']['stages']['ocr']['cuda'] is True
        print('CUDA verified for PyTorch, enhancement, and Paddle.',flush=True)
        benchmark = wait(call('POST','/api/benchmarks',json={'manifest_path':str(labels)}))
        evidence['benchmark'] = benchmark
        print('Benchmark:',json.dumps(benchmark['benchmark']),flush=True)
        image_job = wait(call('POST','/api/jobs/paths',json={'paths':[str(full_image)]}))
        evidence['image'] = image_job
        assert image_job['counts']['detections']>=2,'Expected two real plate detections'
        print('Image detections:',image_job['counts']['detections'],flush=True)
        video_job = wait(call('POST','/api/jobs/paths',json={'paths':[str(video_path)]}))
        detections = call('GET',f'/api/jobs/{video_job["id"]}/detections?limit=200')['items']
        sampled = {d['frame_index'] for d in detections}
        assert sampled == set(range(0,60,6)),sampled
        assert len(detections)>=20, len(detections)
        evidence['video'] = video_job
        evidence['video']['sampled_frames'] = sorted(sampled)
        print('Video detections:',len(detections),'across',len(sampled),'sampled frames',flush=True)
        for kind in ['csv','json','crops','annotated']:
            response = client.get(f'/api/jobs/{video_job["id"]}/export/{kind}')
            response.raise_for_status()
            output = target/f'video-export.{kind if kind in {"csv","json"} else kind+".zip"}'
            output.write_bytes(response.content)
            if kind=='annotated':
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    rendered = target/'annotated.mp4'
                    rendered.write_bytes(archive.read('s00000.mp4'))
                with av.open(str(video_path)) as original,av.open(str(rendered)) as annotated:
                    before = [float(f.time) for f in original.decode(video=0)]
                    after = [float(f.time) for f in annotated.decode(video=0)]
                    assert before==after
            print('Export verified:',kind,flush=True)
        rerun = wait(call('POST',f'/api/jobs/{image_job["id"]}/rerun',json={'stage':'ocr'}))
        assert rerun['counts']==image_job['counts']
        evidence['rerun'] = rerun
    (ROOT/'.runtime'/'real-inference-report.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Real inference verification passed.',flush=True)


if __name__=='__main__':
    main()
