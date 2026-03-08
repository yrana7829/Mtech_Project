import torch
import torch.nn as nn
import torchvision.models as models


class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def get_model(model_name, num_classes):

    if model_name == "resnet18":

        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model

    elif model_name == "mobilenetv2":

        base = models.mobilenet_v2(weights="DEFAULT")
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)

        # MobileNet requires quant wrapper
        model = QuantWrapper(base)

        return model

    else:
        raise ValueError("Unknown model")
