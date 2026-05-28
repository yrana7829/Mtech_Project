import torch
import pandas as pd


class ActivationClippingAnalyzer:

    def __init__(self, target_layers, calibration_ranges):

        self.target_layers = target_layers

        self.calibration_ranges = calibration_ranges

        self.results = {}

        self.handles = []

    def register_hooks(self, model):

        def hook_fn(name):

            def hook(module, input, output):

                activation = output.detach().cpu().flatten()

                calib_min = self.calibration_ranges[name]["min"]

                calib_max = self.calibration_ranges[name]["max"]

                total = len(activation)

                below_clip = (activation < calib_min).sum().item()

                above_clip = (activation > calib_max).sum().item()

                clipped = below_clip + above_clip

                if name not in self.results:

                    self.results[name] = {
                        "total": 0,
                        "clipped": 0,
                        "below_clip": 0,
                        "above_clip": 0,
                    }

                self.results[name]["total"] += total

                self.results[name]["clipped"] += clipped

                self.results[name]["below_clip"] += below_clip

                self.results[name]["above_clip"] += above_clip

            return hook

        for name, module in model.named_modules():

            if name in self.target_layers:

                handle = module.register_forward_hook(hook_fn(name))

                self.handles.append(handle)

                print(f"[INFO] Hook attached: " f"{name}")

    def remove_hooks(self):

        for handle in self.handles:
            handle.remove()

        self.handles = []

    def compute_results(self):

        rows = []

        for layer, stats in self.results.items():

            total = stats["total"]

            clipped_ratio = stats["clipped"] / total

            below_ratio = stats["below_clip"] / total

            above_ratio = stats["above_clip"] / total

            rows.append(
                {
                    "layer": layer,
                    "total_activations": total,
                    "clipping_ratio": clipped_ratio,
                    "below_clip_ratio": below_ratio,
                    "above_clip_ratio": above_ratio,
                }
            )

        return pd.DataFrame(rows)
