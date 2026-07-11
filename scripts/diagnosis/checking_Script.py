import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.models.model_loader import get_model

from src.quantization.proposed.proposed_lps_v2 import apply_proposed_lps_v2
from src.quantization.proposed.proposed_twc_v2 import apply_proposed_twc_v2
from src.quantization.proposed.proposed_mixed_precision_v2 import (
    apply_proposed_mixed_precision_v2,
)

from src.quantization.proposed.proposed_ptq_pipeline_v2 import (
    apply_proposed_ptq_pipeline_v2,
)

model = get_model("mobilenetv2", num_classes=10)

model = apply_proposed_ptq_pipeline_v2(
    model,
    "cpu",
)

print(type(model.features[0][0]))
print(model.features[0][0].ptqpp_bits)
