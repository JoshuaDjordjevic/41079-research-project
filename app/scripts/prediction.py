from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PredictionResult:
    top_predictions: List
    all_probabilities: Dict[str, float]