import torch
import torch.quantization as quant


def naive_ptq(model, calibration_loader, device):

    print("\n===== Starting Naive PTQ =====")

    # PTQ must run on CPU
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    print("Model moved to CPU")

    # Set quantization backend
    torch.backends.quantized.engine = "fbgemm"
    print("Quantization backend set to:", torch.backends.quantized.engine)

    # Assign quantization configuration
    model.qconfig = quant.get_default_qconfig("fbgemm")
    print("QConfig assigned")

    # Prepare model for calibration
    quant.prepare(model, inplace=True)
    print("Model prepared for calibration")

    print("Starting calibration...")

    with torch.no_grad():

        for i, (images, _) in enumerate(calibration_loader):

            images = images.to(device)
            model(images)

            if i % 20 == 0:
                print(f"Calibration batch {i}")

            # stop early (for speed)
            if i == 50:
                break

    print("Calibration completed")

    # Convert to quantized model
    quantized_model = quant.convert(model, inplace=False)

    print("Model converted to INT8")

    print("===== Naive PTQ Complete =====\n")

    return quantized_model
