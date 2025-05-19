import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

from .prediction import PredictionResult

class TomatoClassifier:
    def __init__(self, model_path='models/tomato.h5', class_indices_path='models/tomato_class_indices.json', img_size=224):
        """
        Load a pre-trained tomato disease classification model.

        Args:
            model_path: Path to the .h5 model file
            class_indices_path: Path to the class indices JSON file
            img_size: Size to resize the input images
        """
        self.model = load_model(model_path)
        self.img_size = img_size

        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        self.indices_to_classes = {v: k for k, v in self.class_indices.items()}

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
            (self.indices_to_classes[str(i)], float(predictions[i]))
            for i in top_indices
        ]

        return PredictionResult(
            top_predictions,
            {self.indices_to_classes[str(i)]: float(prob) for i, prob in enumerate(predictions)}
        )
