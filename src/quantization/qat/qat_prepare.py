import copy
import torch
import torchvision
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat


def prepare_mobilenetv2_qat(fp32_model):

    torch.backends.quantized.engine = "fbgemm"

    num_classes = fp32_model.classifier[1].out_features

    # Official quantizable MobileNetV2
    qat_model = torchvision.models.quantization.mobilenet_v2(
        weights=None, quantize=False
    )

    # Replace classifier
    qat_model.classifier[1] = torch.nn.Linear(
        qat_model.classifier[1].in_features, num_classes
    )

    # IMPORTANT:
    # strict=False because quantizable
    # MobileNetV2 has extra wrapper modules
    missing, unexpected = qat_model.load_state_dict(
        fp32_model.state_dict(), strict=False
    )

    print(f"Missing keys: {len(missing)} | " f"Unexpected keys: {len(unexpected)}")

    qat_model.train()

    # Proper fusion
    qat_model.fuse_model(is_qat=True)

    # QAT config
    qat_model.qconfig = get_default_qat_qconfig("fbgemm")

    # Insert fake quant
    qat_model = prepare_qat(qat_model, inplace=False)

    return qat_model
