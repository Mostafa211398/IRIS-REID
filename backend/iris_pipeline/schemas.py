from typing import Literal
from pydantic import BaseModel, Field, model_validator


class JobOptions(BaseModel):
    mode: Literal["vehicle", "crop"] = "vehicle"
    stop_after: Literal["detect", "enhance", "ocr"] = "ocr"
    sample_fps: float = Field(5, gt=0, le=60)
    confidence: float = Field(.4, ge=.01, le=1)
    iou: float = Field(.45, ge=.01, le=1)
    image_size: int = Field(1280, ge=320, le=2048, multiple_of=32)
    padding: int = Field(1, ge=0, le=100)
    upscale: int = Field(3, ge=1, le=6)
    detail_enhancement: bool = True
    review_confidence: float = Field(.8, ge=0, le=1)
    review_quality: float = Field(.36, ge=0, le=1)

    @model_validator(mode="after")
    def crop_stages(self):
        if self.mode == "crop" and self.stop_after == "detect":
            raise ValueError("Plate-crop mode begins at enhancement")
        return self


class PathRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    recursive: bool = False
    options: JobOptions = Field(default_factory=JobOptions)


class RerunRequest(BaseModel):
    stage: Literal["enhance", "ocr"]
    options: JobOptions | None = None


class Correction(BaseModel):
    text: str = Field(max_length=100)
    note: str = Field(default="", max_length=1000)


class BenchmarkRequest(BaseModel):
    manifest_path: str
    options: JobOptions = Field(default_factory=lambda: JobOptions(mode="crop"))
