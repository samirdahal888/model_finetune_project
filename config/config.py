from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    name: str = "distilbert-base-uncased"
    max_length: int = 128
    num_labels: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    num_epoch: int = 1
    learning_rate: float = 2e-5


@dataclass(frozen=True)
class ServerConfig:
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    streamlit_port: int = 8501

    @property
    def api_url(self):
        return f"http://{self.api_host}:{self.api_port}"


class Config:
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    MODEL_SAVE_DIR: Path = PROJECT_ROOT / "models"
    FINE_TUNE_MODEL_PATH: Path = MODEL_SAVE_DIR / "fine-tuned-distilbert"

    # datset
    DATASET_NAME: str = "ag_news"
    LABEL_NAME: List[str] = ["World", "Sports", "Business", "Sci/Tech"]
    NUM_LABELS: int = len(LABEL_NAME)

    _model_config: ModelConfig = ModelConfig()
    _server_config: ServerConfig = ServerConfig()
    _training_config: TrainingConfig = TrainingConfig()

    MODEL_NAME: str = _model_config.name
    MAX_LENGTH: int = _model_config.max_length

    # training settings
    BATCH_SIZE: int = _training_config.batch_size
    LEARNING_RATE: float = _training_config.learning_rate
    NUM_EPOCH: int = _training_config.num_epoch

    # server settings
    API_HOST: str = _server_config.api_host
    API_PORT: int = _server_config.api_port
    API_URL: str = _server_config.api_url
    STREAMLIT_PORT: int = _server_config.streamlit_port
