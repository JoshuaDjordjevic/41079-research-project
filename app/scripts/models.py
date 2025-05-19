import torch
from . import PotatoClassifier
from . import StrawberryClassifier
from . import TomatoClassifier

CLASS_NAMES_POTATO = [
    'Black Scurf', 'Blackleg', 'Common Scab', 'Dry Rot',
    'Healthy', 'Miscellaneous', 'Pink Rot', 'Early Blight',
    'Late Blight', 'Healthy']

CLASS_NAMES_STRAWBERRY = [
    'Angular Leafspot', 'Anthracnose Fruit Rot',
    'Blossom Blight', 'Gray Mold', 'Leaf Spot',
    'Powdery Mildew Fruit', 'Powdery Mildew Leaf']

CLASS_NAMES_TOMATO = [
    'Bacterial Spot', 'Early Blight', 'Late Blight',
    'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites',
    'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus',
    'Healthy', 'Powdery Mildew',]

class ModelManager(object):
    def __init__(self,
                 fp_model_potato:str,
                 fp_model_strawberry:str,
                 fp_model_tomato:str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_potato = PotatoClassifier(self.device, CLASS_NAMES_POTATO, fp_model_potato)
        self.model_strawberry = StrawberryClassifier(CLASS_NAMES_STRAWBERRY, fp_model_strawberry)
        self.model_tomato = TomatoClassifier(CLASS_NAMES_TOMATO, fp_model_tomato)
    
    def predict_potato(self, image_path:str):
        return self.model_potato.predict_image(image_path)

    def predict_strawberry(self, image_path:str):
        return self.model_strawberry.predict_image(image_path)
    
    def predict_tomato(self, image_path:str):
        return self.model_tomato.predict_image(image_path)