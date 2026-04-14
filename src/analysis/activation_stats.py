import torch
import numpy as np


class ActivationStatsCollector:
    def __init__(self):
        self.stats = {}

    def register_hooks(self, model):

        def hook_fn(name):
            def hook(module, input, output):

                if not isinstance(output, torch.Tensor):
                    return

                data = output.detach().cpu().view(-1)

                if data.numel() == 0:
                    return

                data_np = data.numpy()

                stat = {
                    "min": float(np.min(data_np)),
                    "max": float(np.max(data_np)),
                    "mean": float(np.mean(data_np)),
                    "std": float(np.std(data_np)),
                    "p99": float(np.percentile(data_np, 99)),
                    "p999": float(np.percentile(data_np, 99.9)),
                }

                if name not in self.stats:
                    self.stats[name] = []

                self.stats[name].append(stat)

            return hook

        self.hooks = []

        for name, module in model.named_modules():

            # Focus on key layers
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ReLU):
                h = module.register_forward_hook(hook_fn(name))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def aggregate(self):

        final_stats = {}

        for layer, values in self.stats.items():

            aggregated = {}

            for key in values[0].keys():

                aggregated[key] = float(np.mean([v[key] for v in values]))

            final_stats[layer] = aggregated

        return final_stats
