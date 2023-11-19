from torchvision import transforms, datasets
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.RandomRotation(5),
    transforms.Normalize(mean = [0.1728], std = [0.2112]),
])

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.1728], std = [0.2112]),
])

train_dataset = datasets.ImageFolder(
    root = "/home/maheel/Documents/Python/Route_2/Dataset/Training",
    transform = train_transforms)

test_dataset = datasets.ImageFolder(
    root = "/home/maheel/Documents/Python/Route_2/Dataset/Testing",
    transform = test_transforms)

# My poor GPU can only handle a batch_size of 16 :(
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
