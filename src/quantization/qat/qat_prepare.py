import copy
import torch
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat


def prepare_mobilenetv2_qat(model):

    model = copy.deepcopy(model)

    torch.backends.quantized.engine = "fbgemm"

    model.train()

    # attach fuse_model dynamically
    model.fuse_model = lambda: torch.quantization.fuse_modules(model, [], inplace=True)

    # Fuse MobileNetV2 blocks
    for module_name, module in model.named_modules():

        # Conv-BN-ReLU blocks
        if hasattr(module, "0") and hasattr(module, "1"):

            if isinstance(getattr(module, "0"), torch.nn.Conv2d) and isinstance(
                getattr(module, "1"), torch.nn.BatchNorm2d
            ):

                if hasattr(module, "2"):

                    if isinstance(getattr(module, "2"), torch.nn.ReLU6):

                        torch.quantization.fuse_modules(
                            module, ["0", "1", "2"], inplace=True
                        )

                else:

                    torch.quantization.fuse_modules(module, ["0", "1"], inplace=True)

    model.qconfig = get_default_qat_qconfig("fbgemm")

    qat_model = prepare_qat(model, inplace=False)

    return qat_model
