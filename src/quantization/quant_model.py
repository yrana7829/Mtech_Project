import torch
import torch.nn as nn
import torch.quantization as quant


class QuantizedModel(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.quant = quant.QuantStub()
        self.model = model
        self.dequant = quant.DeQuantStub()

    def forward(self, x):

        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x
