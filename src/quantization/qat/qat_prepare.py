import torch
import torch.quantization as tq


def prepare_qat_model(model):

    model.train()

    model.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Fuse modules if applicable
    if hasattr(model, "fuse_model"):
        model.fuse_model()

    tq.prepare_qat(model, inplace=True)

    return model
