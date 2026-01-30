from src.data_loader import DataLoader
from src.model import ModelManager
from src.trainer import ModelTrainer


def create_training_pipeline():
    """
    Execute the complete training pipeline.
    """
    dataloader = DataLoader()

    data = dataloader.prepare_dataset()

    model_manager = ModelManager()

    model, tokenizer = model_manager.create_model()

    trainer = ModelTrainer(model, tokenizer, data)

    trainer.train()


if __name__ == "__main__":
    create_training_pipeline()
