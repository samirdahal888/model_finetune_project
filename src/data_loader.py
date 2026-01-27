# this file should load data and prepare it for the model
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, DatasetDict
from config.config import Config
from typing import Dict, Any


class DataLoader:
    def __init__(self):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME
        )
        self.dataset: DatasetDict = None

    def load_data(self) -> DatasetDict:
        self.dataset = load_dataset(Config.DATASET_NAME)

        return self.dataset

    def tokenized_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
        )

    def prepare_dataset(self) -> DatasetDict:
        if self.dataset is None:
            self.load_data()

        tokenized_data = self.dataset.map(
            self.tokenized_batch, batch_size=Config.BATCH_SIZE, batched=True
        )

        return tokenized_data
