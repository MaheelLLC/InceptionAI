from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model.load_state_dict(torch.load("resnet_model.pth"))
model.eval()

res_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.1728, 0.1728, 0.1728],
                         std = [0.2071, 0.2071, 0.2071])
])

image_path = "/home/maheel/Documents/Python/Route_2/Dataset/Testing/asp_notumor/Te-no_0239.jpg"
image = Image.open(image_path)
image = res_transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

    if predicted == 0:
        print("This image has glioma.")
    elif predicted == 1:
        print("This image has meningioma.")
    else:
        print("This image has no tumors.")