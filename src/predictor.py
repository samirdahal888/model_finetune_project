from dataclasses import dataclass
from src.model import ModelManager
from typing import Dict
import torch
from config.config import Config
from torch.nn.functional import softmax
from logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    text: str
    predicted_label: str
    predicted_id: int
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "predicted_label": self.predicted_label,
            "predicted_id": self.predicted_id,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class Predictor:
    """Handles predictions using the fine-tuned classification model."""

    def __init__(self, model_path: str = None):
        self.model_manager = ModelManager()
        self.model, self.tokenizer = self.model_manager.load_model(model_path)

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text for model inference."""

        logger.debug(f"Tokenizer input text length = {len(text)} ")
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
        )

    def compute_probabilities(self, logits: torch.Tensor) -> Dict[str, float]:
        """Convert model logits to class probabilities."""

        logger.debug("computing softmax probability")
        probs = softmax(logits, dim=-1).squeeze().tolist()
        return {label: round(prob, 4) for label, prob in zip(Config.LABEL_NAME, probs)}

    def predict(self, text: str):
        """
        Classify a single text input.

        Args:
            text: The news article text to classify.

        Returns:
            Dictionary containing prediction results.
        """
        logger.info("Running prediction")
        tokens = self.tokenize(text)
        tokens = {key: val.to(self.model_manager.device) for key, val in tokens.items()}
        logger.debug("Input moved to device: %s", self.model_manager.device)

        with torch.no_grad():
            output = self.model(**tokens)

        logits = output.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        probabilities = self.compute_probabilities(logits=logits)

        predicted_label = Config.LABEL_NAME[predicted_id]
        confidence = max(probabilities.values())
        logger.info(
            "Prediction completed | label=%s | confidence=%.4f",
            predicted_label,
            confidence,
        )

        result = PredictionResult(
            text=text,
            predicted_label=predicted_label,
            predicted_id=predicted_id,
            confidence=confidence,
            probabilities=probabilities,
        )

        return result.to_dict()


if __name__ == "__main__":
    predictor = Predictor()

    sample_texts = [
        "Apple releases new iPhone with revolutionary AI features",
        "Manchester United wins Premier League title",
        "Stock market reaches all-time high amid economic growth",
        "Scientists discover new exoplanet in habitable zone",
    ]

    for text in sample_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['predicted_label']} ({result['confidence']:.1%})")
