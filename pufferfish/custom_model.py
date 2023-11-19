import torch.nn as nn
import torch.nn.functional as F

# Custom model
class Pufferfish(nn.Module):
    def __init__(self):
        super(Pufferfish, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # 256 * 32 * 32 = 262144
        self.fc1 = nn.Linear(262144, 1024)
        self.fc2 = nn.Linear(1024, 3)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)

        x = x.view(-1, 262144)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    