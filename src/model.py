# here we have to load the model that we will used for training
# - i should  define training parameters

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from typing import Tuple, Optional
from config.config import Config
from src.utils import get_device


class ModelManager:
    def __init__(self):
        self.device = get_device()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def create_label_mapping(self):
        id2label = {i: label for i, label in enumerate(Config.LABEL_NAME)}
        label2id = {label: i for i, label in enumerate(Config.LABEL_NAME)}

        return id2label, label2id

    def create_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        id2label, label2id = self.create_label_mapping()
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=Config.NUM_LABELS,
            id2label=id2label,
            label2id=label2id,
        )

        self.model.to(self.device)

        return self.model, self.tokenizer

    def load_model(
        self, model_path: Optional[str] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        path = model_path or Config.FINE_TUNE_MODEL_PATH

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.model.to(self.device)

        return self.model, self.tokenizer
