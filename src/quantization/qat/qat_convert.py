import copy
import torch


def convert_qat_model(model):

    model = copy.deepcopy(model)

    model.cpu()
    model.eval()

    quantized_model = torch.ao.quantization.convert(model, inplace=False)

    return quantized_model
