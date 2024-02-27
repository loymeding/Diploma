from torchvision.models import resnet50
from torch import nn
from typing import List


def get_model(num_classes: int) -> resnet50:
    resnet = resnet50()

    for param in resnet.parameters():
        param.requires_grad = False

    for param in resnet.layer4.parameters():
        param.requires_grad = True
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)

    return resnet
