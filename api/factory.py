from fastapi import FastAPI
from api import router
import os
from config.config import Config
from src.predictor import Predictor
from logger.logger import get_logger
from contextlib import asynccontextmanager


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(str(Config.FINE_TUNE_MODEL_PATH)):
        router.predictor = Predictor()
        logger.info("Model loaded")

    else:
        logger.warning("Model not found")

    yield

    logger.info("Application shutting down")


def create_app() -> FastAPI:
    app = FastAPI(title="Text classification API", lifespan=lifespan)
    app.include_router(router.router)

    @app.get("/")
    def root():
        return {"message": "Text classification api"}

    return app
