"""JSON-lines OCR worker. Paddle is loaded only in its isolated environment."""
import contextlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("GLOG_minloglevel", "2")


def preprocess(image, shape):
    import cv2
    import numpy as np
    channels, height, width = shape
    if channels != 3:
        raise ValueError("Expected BGR three-channel recognition input")
    # PP-OCR inference RecResizeImg: aspect-preserving resize, zero right padding.
    width = int(height * max(width / height, image.shape[1] / image.shape[0]))
    resized_width = min(width, math.ceil(height * image.shape[1] / image.shape[0]))
    resized = cv2.resize(image, (resized_width, height)).astype("float32").transpose(2, 0, 1)
    resized = (resized / 255 - .5) / .5
    padded = np.zeros((channels, height, width), dtype="float32")
    padded[:, :, :resized_width] = resized
    return padded[None]


def decode(output, dictionary):
    import numpy as np
    if output.ndim != 3 or output.shape[-1] != len(dictionary) + 1:
        raise ValueError(f"CTC vocabulary/output mismatch: {output.shape}")
    indices, scores = output[0].argmax(-1), output[0].max(-1)
    selection = np.ones(len(indices), dtype=bool)
    selection[1:] = indices[1:] != indices[:-1]
    selection &= indices != 0
    text = "".join(dictionary[index-1] for index in indices[selection])
    return {"text": text, "confidence": float(scores[selection].mean()) if selection.any() else 0.0}


class Worker:
    def __init__(self):
        self.predictor = None
        self.model_dir = None

    def load(self, model_dir, device):
        import paddle
        import yaml
        from paddle import inference
        model_dir = Path(model_dir)
        meta = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))
        if meta["Global"]["model_name"] != "arabic_PP-OCRv5_mobile_rec" or meta["PostProcess"]["name"] != "CTCLabelDecode":
            raise ValueError("Unsupported OCR export configuration")
        self.dictionary = meta["PostProcess"]["character_dict"]
        self.shape = next(op["RecResizeImg"]["image_shape"] for op in meta["PreProcess"]["transform_ops"] if "RecResizeImg" in op)
        config = inference.Config(str(model_dir / "inference.json"), str(model_dir / "inference.pdiparams"))
        if device.startswith("cuda"):
            if not paddle.device.is_compiled_with_cuda():
                raise RuntimeError("Paddle is not CUDA-enabled")
            index = int(device.split(":")[-1])
            paddle.set_device(f"gpu:{index}")
            config.enable_use_gpu(128, index)
        else:
            paddle.set_device("cpu")
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(2)
        config.disable_mkldnn()
        config.disable_glog_info()
        config.enable_memory_optim()
        config.enable_new_ir()
        self.predictor = inference.create_predictor(config)
        self.model_dir = str(model_dir)
        self.device = device

    def recognize(self, image):
        tensor = preprocess(image, self.shape)
        handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        handle.reshape(tensor.shape)
        handle.copy_from_cpu(tensor)
        self.predictor.run()
        output = self.predictor.get_output_handle(self.predictor.get_output_names()[0]).copy_to_cpu()
        return decode(output, self.dictionary)

    def dispatch(self, request):
        import cv2
        import numpy as np
        import paddle
        if self.predictor is None or self.model_dir != request["model_dir"]:
            self.load(request["model_dir"], request["device"])
        if request["action"] == "health":
            # Both a device tensor probe and real static-model inference.
            result = paddle.matmul(paddle.ones([16, 16]), paddle.ones([16, 16]))
            probe = float(result.mean())
            assert probe == 16
            self.recognize(np.zeros((48, 160, 3), dtype=np.uint8))
            return {"ready": True, "version": paddle.__version__, "device": paddle.device.get_device(),
                    "cuda": self.device.startswith("cuda"), "probe": probe, "model_inference": True}
        if request["action"] == "recognize":
            image = cv2.imdecode(np.fromfile(request["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot decode OCR crop")
            return self.recognize(image)
        raise ValueError("Unknown worker action")


def main():
    worker = Worker()
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = worker.dispatch(json.loads(line))
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
