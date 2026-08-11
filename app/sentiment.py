from functools import lru_cache
from transformers import pipeline

MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

class SentimentAnalyzer:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = self._load(model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load(model_name):
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=256
        )

    def analyze(self, text: str):
        result = self.model(text)[0]
        raw_label = str(result["label"]).lower()
        score = float(result["score"])

        label_map = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }
        label = label_map.get(raw_label, raw_label)

        signed_score = {
            "positive": score,
            "negative": -score,
            "neutral": 0.0
        }.get(label, 0.0)

        return {
            "label": label,
            "score": score,
            "signed_score": signed_score,
            "model": self.model_name
        }
