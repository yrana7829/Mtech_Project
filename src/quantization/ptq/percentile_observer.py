import torch
from torch.ao.quantization.observer import HistogramObserver


class PercentileObserver(HistogramObserver):
    def __init__(self, percentile=99.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percentile = percentile

    def calculate_qparams(self):

        hist = self.histogram

        if hist.sum() == 0:
            return super().calculate_qparams()

        cdf = torch.cumsum(hist, dim=0)
        cdf = cdf / cdf[-1]

        threshold_idx = torch.searchsorted(
            cdf, torch.tensor(self.percentile / 100.0)
        ).item()

        # compute bin width
        bin_width = (self.max_val - self.min_val) / len(hist)

        max_val = self.min_val + bin_width * (threshold_idx + 1)

        # 🔥 IMPORTANT: asymmetric clipping
        min_val = self.min_val

        return self._calculate_qparams(min_val, max_val)
