import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torch import nn


# Now you can use the `models` module to create your model
model = models.resnet18(pretrained=True)

import torch.nn.functional as F

# Load the saved model checkpoint
checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))

# Load the model architecture
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# Load the model state dictionary
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
#model.load_state_dict(checkpoint['state_dict'])

# Set the model to evaluation mode
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the image
image = Image.open('data/3 SEATER SOFA OG.JPG')
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

# Make a prediction
with torch.no_grad():
    logits = model(image_tensor)
    probas = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probas).item()
    
print(predicted_class)

# Define the label mapping
label_map = {0:"Sofa", 1:"Bed", 2:"Chair"}

# Get the class label from the predicted label index using the label mapping
predicted_class = label_map[predicted_class]
print(predicted_class)

# Load the image
image = Image.open('data/chair2.jpg')
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

# Make a prediction
with torch.no_grad():
    logits = model(image_tensor)
    probas = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probas).item()
    
print(predicted_class)
# Get the class label from the predicted label index using the label mapping
predicted_class = label_map[predicted_class]
print(predicted_class)

