import torch

import torch.utils.data
import torchvision.models
import torch.utils.data

import torch.optim as optim

from tqdm import tqdm
from typing import Callable


def train(model: torchvision.models,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer,
          metric_fn: Callable,
          num_epochs: int = 10,
          device: str = 'cpu') -> tuple[list[float], list[float]]:

    train_metrics = []
    val_metrics = []
    optimizer = optim.Adam(model.parameters(), 0.001)
    model.to(device)
    print('-------START TRAINING-------')
    for epoch in tqdm(range(num_epochs)):
        # Train phase
        model.train()
        train_loss = 0.0
        train_metric = 0.0
        total_train_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_metric += metric_fn(outputs, targets)
            total_train_samples += inputs.size(0)

        train_loss /= total_train_samples
        train_metric /= len(train_loader)
        train_metrics.append(train_metric)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_metric += metric_fn(outputs, targets)
                total_val_samples += inputs.size(0)

        val_loss /= total_val_samples
        val_metric /= len(val_loader)
        val_metrics.append(val_metric)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
               f'Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}')

    return train_metrics, val_metrics
