import torch
import torch.quantization as quant


def naive_ptq(model, calibration_loader, device):

    # PTQ runs on CPU
    device = torch.device("cpu")
    model.to(device)

    model.eval()

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
