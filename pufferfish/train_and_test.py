import torch
import torch.optim as optim
import torch.nn as nn
from custom_model import Pufferfish
from data_process import train_loader, test_loader

# torch.cuda.empty_cache()
# Custom model
model = Pufferfish()

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Load previous model
model.load_state_dict(torch.load("cus_model.pth", map_location=device))

def train_loop(dataloader, model, loss_fn, optimizer):
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

def test_loop(dataloader, model, loss_fn):
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
    
epochs = 1

for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)

print("Done")

torch.save(model.state_dict(), "cus_model.pth")

# Requires a total of 15 epochs to train
# cus_model.pth
# Percent Accuracy: 95-96%

# new tactic: use 1 epoch until you hit 96%
