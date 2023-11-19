import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from res_data_process import res_train_loader, res_test_loader

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Load here
model.load_state_dict(torch.load("resnet_model.pth", map_location=device))

def res_train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 30 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def res_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, correct = 0, 0
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss : " + 
          f"{loss:>8f} \n")
    
epochs = 3

for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------")
    res_train_loop(res_train_loader, model, loss_fn, optimizer)
    res_test_loop(res_test_loader, model, loss_fn)

print("Done")

torch.save(model.state_dict(), "resnet_model.pth")

# Requires a total of 8-9 epochs to train
# resnet_model.pth
# Percent Accuracy: 97%

# new tactic: use single epochs until you hit 97
