import torch
import torch.nn as nn
import torch.nn.functional as F

class TempModel(nn.Module):
    def __init__(self):
        super(TempModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        return x

# Create a temporary model instance and a dummy input tensor
temp_model = TempModel()
temp_input = torch.zeros(1, 1, 512, 512)  # Assuming single-channel 512x512 input

# Forward pass of the dummy input through the temporary model
output = temp_model(temp_input)

# Calculate and print out the total number of features
total_features = int(torch.prod(torch.tensor(output.size()[1:])))
print(f"Total features before the first fully connected layer: {total_features}")