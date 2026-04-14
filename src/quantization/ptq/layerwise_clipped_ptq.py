import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig

from src.quantization.ptq.percentile_observer import PercentileObserver


def build_qconfig_mapping(clipped_layers, percentile=99.0):

    default_qconfig = get_default_qconfig("fbgemm")

    clipped_qconfig = torch.ao.quantization.QConfig(
        activation=PercentileObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine, percentile=percentile
        ),
        weight=torch.ao.quantization.default_per_channel_weight_observer,
    )

    qconfig_mapping = QConfigMapping().set_global(default_qconfig)

    for layer_name in clipped_layers:
        qconfig_mapping = qconfig_mapping.set_module_name(layer_name, clipped_qconfig)

    return qconfig_mapping


def layerwise_clipped_ptq_fx(
    model, calibration_loader, clipped_layers, percentile=99.0
):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    print(f"Clipping applied to {len(clipped_layers)} layers")

    qconfig_mapping = build_qconfig_mapping(clipped_layers, percentile)

    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("Preparing FX quantization...")
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)

    print("Running calibration...")
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            prepared_model(images)

    print("Converting to INT8...")
    quantized_model = convert_fx(prepared_model)

    return quantized_model
