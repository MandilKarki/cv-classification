import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torch import nn
import torch.nn.functional as F

LABEL_MAP = {0: "Sofa", 1: "Bed", 2: "Chair"}


def load_model(model_path):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_class(image_path, model, transform):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(image_tensor)
        probas = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probas).item()
    return predicted_class


def get_class_label(predicted_class):
    return LABEL_MAP.get(predicted_class, "Unknown")


if __name__ == '__main__':
    model_path = 'model.pth'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model = load_model(model_path)
    
    image_path = 'data/3 SEATER SOFA OG.JPG'
    predicted_class = predict_class(image_path, model, transform)
    class_label = get_class_label(predicted_class)
    print(f"Image '{image_path}' predicted as: {class_label}")
    
    image_path = 'data/chair2.jpg'
    predicted_class = predict_class(image_path, model, transform)
    class_label = get_class_label(predicted_class)
    print(f"Image '{image_path}' predicted as: {class_label}")
