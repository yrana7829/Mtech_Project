import copy
import torch
import torchvision


def prepare_mobilenetv2_qat(model):

    model = copy.deepcopy(model)

    model.train()

    # backend
    torch.backends.quantized.engine = "fbgemm"

    # fuse Conv+BN+ReLU
    model_fused = torchvision.models.quantization.mobilenet_v2(
        weights=None, quantize=False
    )

    # replace classifier
    num_classes = model.classifier[1].out_features
    model_fused.classifier[1] = torch.nn.Linear(
        model_fused.classifier[1].in_features, num_classes
    )

    # load FP32 weights
    model_fused.load_state_dict(model.state_dict())

    # fuse internally
    model_fused.fuse_model(is_qat=True)

    # QAT config
    model_fused.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")

    # prepare QAT
    qat_model = torch.ao.quantization.prepare_qat(model_fused, inplace=False)

    return qat_model
