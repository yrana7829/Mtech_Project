import torch

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


def apply_full_ptq(model, calibration_loader, backend="fbgemm"):

    model.eval()
    model = model.cpu()

    torch.backends.quantized.engine = backend

    qconfig_mapping = get_default_qconfig_mapping(backend)

    example_inputs = next(iter(calibration_loader))[0][:1].cpu()

    print("\n[INFO] Preparing FX PTQ model...")

    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)

    print("[INFO] Running calibration...")

    with torch.no_grad():

        for images, _ in calibration_loader:
            prepared_model(images.cpu())

    print("[INFO] Converting quantized model...")

    quantized_model = convert_fx(prepared_model)

    return quantized_model
