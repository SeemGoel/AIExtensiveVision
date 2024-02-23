import torch
import torchvision.transforms as transforms
from torchvision import datasets

def load_mnist_data(data_dir='./data'):
  batch_size = 512
  train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
  # Test data transformations
  test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1407,), (0.4081,))
    ])
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  test_data = datasets.MNIST('../data', train=False, download=True, transform=train_transforms)
  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
  test_loader = torch.utils.data.DataLoader(test_data, **kwargs)


def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    """Trains the model on the given data."""
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    """Evaluates the model on the test data."""
    model.eval()

  