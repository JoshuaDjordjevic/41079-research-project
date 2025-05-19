import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from typing import List, Tuple, Dict

from .prediction import PredictionResult

class StrawberryClassifier(object):
    def __init__(self, class_names: List[str], model_path: str = "./models/strawberry.h5", img_size: int = 224):
        self.class_names = class_names
        self.model_path = model_path
        self.img_size = img_size
        self.model = self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        return tf.keras.models.load_model(self.model_path)

    def predict_image(self, image_path: str, topk: int = 4) -> PredictionResult:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = predictions.argsort()[-topk:][::-1]
        top_predictions = [(self.class_names[i], float(predictions[i])) for i in top_indices]
        all_probabilities = {self.class_names[i]: float(prob) for i, prob in enumerate(predictions)}

        return PredictionResult(top_predictions, all_probabilities)