import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_mnist_dataloader(root='./data', train=True, download=True):
    dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

