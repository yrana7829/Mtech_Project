import torch.nn as nn
from torchvision import models


def get_model(model_name, num_classes):

    if model_name == "resnet18":

        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenetv2":

        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError("Unknown model")

    return model
