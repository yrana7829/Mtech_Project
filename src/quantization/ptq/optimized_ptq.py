import torch
import torch.nn as nn
import torch.quantization as quant


def cross_layer_equalization(model):

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            w = module.weight.data

            max_per_channel = w.abs().view(w.size(0), -1).max(dim=1)[0]

            scale = max_per_channel.mean() / (max_per_channel + 1e-8)

            scale = scale.view(-1, 1, 1, 1)

            module.weight.data *= scale

    return model


def optimized_ptq(model, calibration_loader):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    torch.backends.quantized.engine = "fbgemm"

    print("Applying BN folding...")

    if hasattr(model, "fuse_model"):
        model.fuse_model()

    print("Applying Cross Layer Equalization...")

    model = cross_layer_equalization(model)

    model.qconfig = quant.get_default_qconfig("fbgemm")

    quant.prepare(model, inplace=True)

    print("Running calibration...")

    with torch.no_grad():

        for i, (images, _) in enumerate(calibration_loader):

            model(images)

            if i > 50:
                break

    print("Converting to INT8...")

    quantized_model = quant.convert(model, inplace=False)

    return quantized_model
