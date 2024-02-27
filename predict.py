import torch


def predict(model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            device:str = 'cpu') -> list[float]:
    print("-------Start predicting-------")
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(
                outputs.cpu().numpy())

    return predictions
