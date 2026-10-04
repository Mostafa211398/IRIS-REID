import uvicorn
from .config import settings

uvicorn.run("iris_pipeline.app:app", host=settings.host, port=settings.port)
