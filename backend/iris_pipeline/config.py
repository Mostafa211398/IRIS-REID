from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRIS_", env_file=ROOT / ".env", extra="ignore")
    root: Path = ROOT
    host: str = "127.0.0.1"
    port: int = 8014
    frontend_port: int = 5187
    device: str = "auto"
    ocr_timeout: float = 120
    annotation_font: str = ""

    @property
    def data(self):
        return self.root / ".data"

    @property
    def paddle_python(self):
        return self.root / ".runtime" / "paddle" / ("Scripts/python.exe" if __import__("os").name == "nt" else "bin/python")

    def prepare(self):
        for folder in (self.data, self.data / "jobs", self.data / "models", self.root / ".runtime"):
            folder.mkdir(parents=True, exist_ok=True)


settings = Settings()
