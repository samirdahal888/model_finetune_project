from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from datasets import DatasetDict
from config.config import Config
from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import wandb


class ModelTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: DatasetDict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.trainer = None

    def create_training_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            num_train_epochs=Config.NUM_EPOCH,
            output_dir=str(Config.FINE_TUNE_MODEL_PATH),
            learning_rate=Config.LEARNING_RATE,
            load_best_model_at_end=True,
            per_device_train_batch_size=Config.BATCH_SIZE,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="accuracy",
            report_to="wandb",
        )

    def compute_metrix(self, eval_pred: Tuple):
        logits, lable = eval_pred
        prediction = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(lable, prediction)
        return {"accuracy": accuracy}

    def create_trainer(self) -> Trainer:
        training_args = self.create_training_arguments()

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            compute_metrics=self.compute_metrix,
        )

    def train(self):
        wandb.init(
            project="distilbert-ag-news",
            name="distilbert-base-uncased",
            config={
                "epochs": Config.NUM_EPOCH,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LEARNING_RATE,
            },
        )
        self.trainer = self.create_trainer()

        train_result = self.trainer.train()
        self.save_model()
        wandb.finish()
        return train_result

    def save_model(self) -> None:
        """Save the trained model and tokenizer."""
        self.trainer.save_model()
        self.tokenizer.save_pretrained(str(Config.FINE_TUNE_MODEL_PATH))
        print(f"Model saved to: {Config.FINE_TUNE_MODEL_PATH}")
