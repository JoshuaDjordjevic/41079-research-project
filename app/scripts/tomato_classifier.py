import os
from typing import List
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

from .prediction import PredictionResult

class TomatoClassifier:
    def __init__(self, class_names: List[str], model_path='models/tomato.h5', img_size=224):
        """
        Load a pre-trained tomato disease classification model.

        Args:
            model_path: Path to the .h5 model file
            class_indices_path: Path to the class indices JSON file
            img_size: Size to resize the input images
        """
        self.class_names = class_names
        self.model = load_model(model_path)
        self.img_size = img_size

    def predict_image(self, image_path) -> PredictionResult:
        """
        Predict the disease class for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with prediction probabilities and top predictions
        """
        img = image.load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = self.model.predict(img_array)[0]

        top_indices = predictions.argsort()[-4:][::-1]
        top_predictions = [
            (self.class_names[i], float(predictions[i]))
            for i in top_indices
        ]

        return PredictionResult(
            top_predictions,
            {self.class_names[i]: float(prob) for i, prob in enumerate(predictions)}
        )
