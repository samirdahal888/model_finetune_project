# this file should load data and prepare it for the model
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, DatasetDict
from config.config import Config
from typing import Dict, Any
from logger.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Manages dataset loading and preprocessing for text classification.

    Responsibilities:
        - Download and cache the AG News dataset
        - Tokenize text data for model consumption
        - Prepare data in PyTorch-compatible format
    """

    def __init__(self):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME
        )
        self.dataset: DatasetDict = None

    def load_data(self) -> DatasetDict:
        """
        Load the AG News dataset from Hugging Face Hub.

        Returns:
            DatasetDict containing 'train' and 'test' splits.
        """
        logger.info("Starting dataset loading")
        logger.info(f"Dataset name{Config.DATASET_NAME}")

        self.dataset = load_dataset(Config.DATASET_NAME)

        return self.dataset

    def tokenized_batch(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a batch of text examples.

        Args:
            examples: Batch of examples with 'text' field.

        Returns:
            Tokenized batch with 'input_ids' and 'attention_mask'.
        """
        logger.debug(f"Tokenizing batch with {len(examples['text'])} samples ")
        logger.info("")
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
        )

    def prepare_dataset(self) -> DatasetDict:
        """
        Load and prepare the complete dataset for training.

        Returns:
            Tokenized dataset ready for PyTorch training.
        """
        if self.dataset is None:
            logger.info("Dataset not loaded yet ,loading dataset")
            self.load_data()

        tokenized_data = self.dataset.map(
            self.tokenized_batch, batch_size=Config.BATCH_SIZE, batched=True
        )
        logger.info("Dataset Preparation completed")

        return tokenized_data
