import torch
import numpy as np
from tqdm import tqdm


def predict_proba(model: torch.nn.Module,
                  test_loader: torch.utils.data.DataLoader,
                  device: str = 'cpu') -> np.array:
    print("-------START PREDICTING-------")
    model.eval()
    predictions = []
    for x_batch, y_batch in tqdm(test_loader):
        if device == 'cuda':
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        y_pred = model(x_batch)
        predictions.append(y_pred.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    return predictions


def predict(model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            device: str = 'cpu') -> np.array:
    predictions = predict_proba(model, test_loader, device)
    predictions = predictions.argmax(axis=1)
    return predictions
