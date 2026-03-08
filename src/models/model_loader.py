import torch.nn as nn
import torchvision.models as models


def get_model(model_name, num_classes):

    if model_name == "resnet18":

        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    elif model_name == "mobilenetv2":

        model = models.mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

        return model

    else:
        raise ValueError("Unknown model")
