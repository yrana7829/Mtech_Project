import torch
import torch.quantization as quant
from src.quantization.quant_model import QuantizedModel


def naive_ptq(model, calibration_loader, device):

    # PTQ runs on CPU
    model = QuantizedModel(model)
    model.eval()
    model.to(device)

    # Set backend
    torch.backends.quantized.engine = "fbgemm"

    # Assign quantization config
    model.qconfig = quant.get_default_qconfig("fbgemm")

    # Prepare model
    quant.prepare(model, inplace=True)

    # Calibration pass
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            model(images)

    # Convert to quantized model
    quantized_model = quant.convert(model, inplace=False)

    return quantized_model
