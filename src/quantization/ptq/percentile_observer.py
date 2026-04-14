import torch
from torch.ao.quantization.observer import HistogramObserver


class PercentileObserver(HistogramObserver):
    def __init__(self, percentile=99.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percentile = percentile

    def calculate_qparams(self):

        # Use histogram + min/max to approximate distribution
        hist = self.histogram

        if hist.sum() == 0:
            return super().calculate_qparams()

        # Compute cumulative distribution
        cdf = torch.cumsum(hist, dim=0)
        cdf = cdf / cdf[-1]

        # Find percentile index
        threshold_idx = torch.searchsorted(
            cdf, torch.tensor(self.percentile / 100.0)
        ).item()

        # Map index → value using min/max range
        bin_width = (self.max_val - self.min_val) / len(hist)

        max_val = self.min_val + bin_width * (threshold_idx + 1)
        min_val = -max_val  # symmetric clipping

        return self._calculate_qparams(min_val, max_val)
