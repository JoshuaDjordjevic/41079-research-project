from typing import List
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

from .prediction import PredictionResult

class PotatoClassifier(object):
    def __init__(self, device: torch.device, class_names: List[str], model_path: str = "./models/potato_disease_classifier.pth"):
        self.device = device
        self.class_names = class_names
        self.model = self.get_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_model(self):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(self.class_names))
        )

        return model.to(self.device)

    def predict_image(self, image_path: str, img_size: int = 224, topk: int = 4) -> PredictionResult:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, k=topk)

        top_predictions = [
            (self.class_names[top_indices[i].item()], float(top_probs[i].item()))
            for i in range(topk)
        ]

        all_probabilities = {
            self.class_names[i]: float(probabilities[i].item())
            for i in range(len(self.class_names))
        }

        return PredictionResult(top_predictions, all_probabilities)