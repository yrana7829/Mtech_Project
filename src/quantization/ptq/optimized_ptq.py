import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


def cross_layer_equalization(model):

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            w = module.weight.data

            max_per_channel = w.abs().view(w.size(0), -1).max(dim=1)[0]

            scale = max_per_channel.mean() / (max_per_channel + 1e-8)

            scale = scale.view(-1, 1, 1, 1)

            module.weight.data *= scale

    return model


def optimized_ptq_fx(model, calibration_loader):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    print("Applying BN fusion...")
    if hasattr(model, "fuse_model"):
        model.fuse_model()

    # print("Applying Cross Layer Equalization...")
    # model = cross_layer_equalization(model)

    # FX quantization setup
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}

    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("Preparing FX quantization...")
    prepared_model = prepare_fx(model, qconfig_dict, example_inputs)

    print("Running calibration on fixed subset...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            prepared_model(images)

    print("Converting to INT8...")
    quantized_model = convert_fx(prepared_model)

    return quantized_model
