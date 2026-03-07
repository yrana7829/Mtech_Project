import torch
import torch.quantization as quant


def naive_ptq(model, calibration_loader, device):

    model.eval()
    model.to(device)

    # assign quantization configuration
    model.qconfig = quant.get_default_qconfig("fbgemm")

    # prepare model for calibration
    quant.prepare(model, inplace=True)

    # calibration pass
    with torch.no_grad():
        for images, _ in calibration_loader:
            images = images.to(device)
            model(images)

    # convert to quantized model
    quantized_model = quant.convert(model, inplace=True)

    return quantized_model
