from flask import Flask, request, render_template
from torchvision import transforms, models
from aspect_ratio import resize_and_pad
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os

# Since pufferfish is another directory
import sys
sys.path.append("../pufferfish")
from custom_model import Pufferfish

app = Flask(__name__, template_folder='templates', static_url_path='/static')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# custom model: pufferfish
pufferfish = Pufferfish()
pufferfish.load_state_dict(torch.load("../pufferfish/cus_model.pth"))
pufferfish.eval()

# resnet model: resnet
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 3)
resnet.load_state_dict(torch.load("../resnet/resnet_model.pth"))
resnet.eval()

# pufferfish data processing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.1728], std = [0.2112]),
])

# resnet data processing
res_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.1728, 0.1728, 0.1728],
                         std = [0.2071, 0.2071, 0.2071])
])

# Let's make the view function (combine url, template, and python renderings
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file imported'
        
        file = request.files['file']

        if file.filename == '':
            return 'No selected file'
        
        if file:
            try:
                # original image
                original_image = Image.open(file.stream)
                img_array = resize_and_pad(np.array(original_image))
                processed_image = Image.fromarray(img_array)

                # Save the processed image to the uploads folder
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                processed_image.save(save_path)

                # Path relative to the static folder to pass to the template
                image_path = os.path.join('uploads', filename)

                # pufferfish prediction
                image = transform(processed_image)
                image = image.unsqueeze(0)
                prediction = predict(image)

                # resnet prediction
                res_image = res_transform(processed_image)
                res_image = res_image.unsqueeze(0)
                res_prediction = res_predict(res_image)

                return render_template('result.html', prediction = prediction, 
                                       res_prediction = res_prediction,
                                       image_path = image_path)
            except IOError:
                return "Invalid image file"
    return render_template('upload.html')


def predict(image):
    with torch.no_grad():
        output = pufferfish(image)
        _, predicted = torch.max(output, 1)

        if predicted == 0:
            return "This image has glioma."
        elif predicted == 1:
            return "This image has meningioma."
        else:
            return "This image has no tumors."
        
def res_predict(image):
    with torch.no_grad():
        output = resnet(image)
        _, predicted = torch.max(output, 1)

        if predicted == 0:
            return "This image has glioma."
        elif predicted == 1:
            return "This image has meningioma."
        else:
            return "This image has no tumors."
