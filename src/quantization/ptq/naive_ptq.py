import torch
import torch.quantization as quant


def naive_ptq(model, calibration_loader, device):

    print("\nStarting Naive PTQ")

    device = torch.device("cpu")
    model.eval()
    model.to(device)

    # Required backend
    torch.backends.quantized.engine = "fbgemm"

    # IMPORTANT: fuse layers for ResNet
    if hasattr(model, "fuse_model"):
        model.fuse_model()

    # Attach qconfig
    model.qconfig = quant.get_default_qconfig("fbgemm")

    print("Preparing model for calibration...")
    quant.prepare(model, inplace=True)

    # Calibration
    print("Running calibration...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            images = images.to(device)
            model(images)

            if i > 50:  # limit calibration batches
                break

    print("Converting to quantized model...")
    quantized_model = quant.convert(model, inplace=False)

    print("PTQ complete\n")

    return quantized_model
