from dataloaders import get_dataloaders
from model import get_model
from train import train
from predict import predict

import torch

import torch.optim as optim
import torch.nn as nn


def accuracy_fn(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


batch_size = 32
shuffle = False
train_dir = '.\\dataset\\plant-seedlings-classification\\train'
test_dir = '.\\dataset\\plant-seedlings-classification\\test'
valid_dir = '.\\dataset\\plant-seedlings-classification\\valid'
lr = 0.001
num_classes = 12
num_epochs = 1
device = 'cpu'

train_loader, valid_loader, test_loader = get_dataloaders(batch_size, shuffle, train_dir, test_dir, valid_dir)
model = get_model(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

train_metrics, val_metrics = train(model, train_loader, valid_loader, criterion, optimizer, accuracy_fn, num_epochs,
                                   device)

predictions = predict(model, test_loader, device)
