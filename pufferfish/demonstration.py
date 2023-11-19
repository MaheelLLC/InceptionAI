from torchvision import transforms
from PIL import Image
import torch

from custom_model import Pufferfish

model = Pufferfish()

model.load_state_dict(torch.load("cus_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.1728], std = [0.2112]),
])

image_path = "/home/maheel/Documents/Python/Route_2/Dataset/Testing/asp_glioma/Te-gl_0080.jpg"
image = Image.open(image_path)
image = transform(image)
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

