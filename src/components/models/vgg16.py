from torchvision.models import vgg16
from torch import nn


def get_model(num_classes: int) -> vgg16:
    print('---------GET MODEL--------')
    vgg = vgg16(pretrained=True)

    for param in vgg.parameters():
        param.requires_grad = False

    num_features = vgg.classifier[-1].in_features
    vgg.classifier[-1] = nn.Linear(num_features, num_classes)

    return vgg
