from dataclasses import dataclass
from src.model import ModelManager
from typing import Dict
import torch
from config.config import Config
from torch.nn.functional import softmax


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
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class Predictor:
    def __init__(self, model_path: str = None):
        self.model_manager = ModelManager()
        self.model, self.tokenizer = self.model_manager.load_model(model_path)

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors="pt",
        )

    def compute_probabilities(self, logits: torch.Tensor) -> Dict[str, float]:
        probs = softmax(logits, dim=-1).squeeze().tolist()
        return {label: round(prob, 4) for label, prob in zip(Config.LABEL_NAME, probs)}

    def predict(self, text: str):
        tokens = self.tokenize(text)
        tokens = {key: val.to(self.model_manager.device) for key, val in tokens.items()}

        with torch.no_grad():
            output = self.model(**tokens)

        logits = output.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        probabilities = self.compute_probabilities(logits=logits)

        result = PredictionResult(
            text=text,
            predicted_label=Config.LABEL_NAME[predicted_id],
            predicted_id=predicted_id,
            confidence=max(probabilities.values()),
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
