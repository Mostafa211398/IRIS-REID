"""Dedicated stage; calls the supplied algorithm exactly once per execution."""
import importlib.util


class Enhancer:
    def __init__(self, root, device):
        spec = importlib.util.spec_from_file_location("iris_reference_enhancement", root / "plate_enhancement.py")
        self.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reference)
        self.device = device

    def enhance(self, image, options):
        sharpness, edges = self.reference.character_metrics(image, self.device)
        metrics = {"char_sharpness": sharpness, "edge_strength": edges}
        applied = options.detail_enhancement and sharpness >= 55 and edges >= 11
        output = self.reference.enhance_crop(image, char_sharpness=sharpness if options.detail_enhancement else 0,
                                             edge_strength=edges if options.detail_enhancement else 0,
                                             upscale=options.upscale, device=self.device)
        return output, {**metrics, "upscale": options.upscale, "detail_enhancement": options.detail_enhancement,
                        "method": "contrast_and_sharpening" if applied else "enlargement_only"}
