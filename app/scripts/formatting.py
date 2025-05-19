from scripts.prediction import PredictionResult


def format_prediction(prediction: PredictionResult) -> str:
    top_3 = prediction.top_predictions[:3]
    return '\n'.join([
        f'{percentage*100:.0f}% chance of "{cls}"' for (cls, percentage) in top_3
    ])