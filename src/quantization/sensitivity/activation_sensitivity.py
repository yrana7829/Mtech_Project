import torch
import torch.nn as nn


class FakeActivationQuantizer(nn.Module):
    """
    Simulates INT8 activation quantization.
    """

    def __init__(self, num_bits=8):
        super().__init__()

        self.num_bits = num_bits
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

    def forward(self, x):

        x_min = x.min()
        x_max = x.max()

        # Avoid divide-by-zero
        if x_max - x_min < 1e-8:
            return x

        scale = (x_max - x_min) / (self.qmax - self.qmin)
        zero_point = self.qmin - x_min / scale

        q_x = torch.clamp(torch.round(x / scale + zero_point), self.qmin, self.qmax)

        dq_x = (q_x - zero_point) * scale

        return dq_x


class ActivationSensitivityAnalyzer:

    def __init__(self, model, target_module_names):

        self.model = model
        self.target_module_names = target_module_names

        self.handles = []

        self.quantizer = FakeActivationQuantizer()

    def _hook_fn(self, module, input, output):

        return self.quantizer(output)

    def register_hooks(self):

        for name, module in self.model.named_modules():

            if name in self.target_module_names:

                handle = module.register_forward_hook(self._hook_fn)

                self.handles.append(handle)

                print(f"[INFO] Hook attached to: {name}")

    def remove_hooks(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []
