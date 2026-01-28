# here we have to load the model that we will used for training
# - i should  define training parameters

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from typing import Tuple, Optional, Dict
from config.config import Config
from src.utils import get_device
from logger.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self):
        self.device = get_device()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def create_label_mapping(self) -> Tuple[Dict]:
        logger.info(
            f"Creating label mapping for the {len(Config.LABEL_NAME)} labels , {Config.LABEL_NAME}"
        )
        id2label = {i: label for i, label in enumerate(Config.LABEL_NAME)}
        label2id = {label: i for i, label in enumerate(Config.LABEL_NAME)}
        logger.debug(f"id2label is : {id2label}")
        logger.debug(f"label2id is : {label2id}")

        return id2label, label2id

    def create_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        logger.info("Initilizing model and tokenizer")
        logger.info(f"The base model is {Config.MODEL_NAME}")
        id2label, label2id = self.create_label_mapping()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=Config.NUM_LABELS,
            id2label=id2label,
            label2id=label2id,
        )
        logger.info(f"Model initilized with the {Config.NUM_LABELS} labels ")

        self.model.to(self.device)
        logger.info(f"Model move to the {self.device}")

        return self.model, self.tokenizer

    def load_model(
        self, model_path: Optional[str] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        path = model_path or Config.FINE_TUNE_MODEL_PATH

        logger.info(f"Fine-tuned model loading from the path: {path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.model.to(self.device)
        logger.info(f"Model move to the {self.device}")

        return self.model, self.tokenizer
