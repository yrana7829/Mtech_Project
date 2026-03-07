from torchvision.models.quantization import resnet18
from torchvision.models import mobilenet_v2


def get_model(model_name, num_classes):

    if model_name == "resnet18":

        model = resnet18(weights="DEFAULT", quantize=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenetv2":

        model = mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    else:
        raise ValueError("Unknown model")

    return model
