import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(
    root = "/home/maheel/Documents/Python/Route_2/Dataset/Training",
    transform = transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim = [0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim = [0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std

mean, std = get_mean_std(dataloader)
print(f"Mean: {mean}, Std: {std}")