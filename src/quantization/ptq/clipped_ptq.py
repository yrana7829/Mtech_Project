import torch
import torch.nn as nn

from torch.ao.quantization import QConfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from src.quantization.ptq.percentile_observer import PercentileObserver


# -------------------------------
# Build percentile-based qconfig
# -------------------------------
def get_percentile_qconfig(percentile=99.9):

    return QConfig(
        activation=PercentileObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine, percentile=percentile
        ),
        weight=torch.ao.quantization.default_per_channel_weight_observer,
    )


# -------------------------------
# FX PTQ with percentile clipping
# -------------------------------
def clipped_ptq_fx(model, calibration_loader, percentile=99.9):

    device = torch.device("cpu")

    model.eval()
    model.to(device)

    # -------------------------------
    # Optional: BN fusion (safe)
    # -------------------------------
    if hasattr(model, "fuse_model"):
        print("Applying BN fusion...")
        model.fuse_model()

    # -------------------------------
    # Set percentile-based qconfig
    # -------------------------------
    print(f"Using percentile observer (p={percentile})...")

    qconfig = get_percentile_qconfig(percentile)
    qconfig_dict = {"": qconfig}

    # -------------------------------
    # FX prepare
    # -------------------------------
    example_inputs = next(iter(calibration_loader))[0][:1].to(device)

    print("Preparing FX quantization...")
    prepared_model = prepare_fx(model, qconfig_dict, example_inputs)

    # -------------------------------
    # Calibration
    # -------------------------------
    print("Running calibration...")

    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            prepared_model(images)

    # -------------------------------
    # Convert to quantized model
    # -------------------------------
    print("Converting to INT8...")
    quantized_model = convert_fx(prepared_model)

    return quantized_model
