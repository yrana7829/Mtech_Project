import torch

from .proposed_lps_v2 import apply_proposed_lps_v2
from .proposed_twc_v2 import apply_proposed_twc_v2
from .proposed_mixed_precision_v2 import apply_proposed_mixed_precision_v2


def apply_proposed_ptq_pipeline_v2(model, device):
    """
    ============================================================
                    PTQ++ v2 PIPELINE
    ============================================================

    Stage 1 : Learned Pre-Scaling (LPS)
    Stage 2 : Tail Weighted Clipping (TWC)
    Stage 3 : Mixed Precision Allocation

    Output:
        Weight-conditioned FP32 model.

    NOTE:
        This function performs ONLY preprocessing.
        Real INT8 quantization is performed afterwards
        using FX PTQ.
    """

    print("\n")
    print("=" * 60)
    print("Starting PTQ++ v2 Pipeline")
    print("=" * 60)

    # ----------------------------------------------------------
    # Stage 1
    # ----------------------------------------------------------
    # model = apply_proposed_lps_v2(model, device)

    # ----------------------------------------------------------
    # Stage 2
    # ----------------------------------------------------------
    model = apply_proposed_twc_v2(model)

    # ----------------------------------------------------------
    # Stage 3
    # ----------------------------------------------------------
    model = apply_proposed_mixed_precision_v2(model)

    print("=" * 60)
    print("PTQ++ v2 preprocessing completed.")
    print("=" * 60)

    return model
