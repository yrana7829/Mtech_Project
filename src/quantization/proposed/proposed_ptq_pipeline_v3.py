import torch

from .proposed_lps_v3 import apply_proposed_lps_v3
from .proposed_twc_v3 import apply_proposed_twc_v3
from .proposed_mpa_v3 import apply_proposed_mpa_v3


def apply_proposed_ptq_pipeline_v3(
    model,
    calibration_loader,
    device,
):
    """
    PTQ++ v3 Pipeline

    Stage 1
        Activation-aware LPS

    Stage 2
        Layer-wise Tail Weighted Clipping

    Stage 3
        Sensitivity-driven Mixed Precision
    """

    print("\n========================================")
    print("Starting PTQ++ v3 Pipeline")
    print("========================================")

    # ---------------------------------------
    # Stage 1
    # ---------------------------------------

    model = apply_proposed_lps_v3(
        model,
        calibration_loader,
        device,
    )

    # ---------------------------------------
    # Stage 2
    # ---------------------------------------

    # model = apply_proposed_twc_v3(model)

    # ---------------------------------------
    # Stage 3
    # ---------------------------------------

    # model = apply_proposed_mpa_v3(model)

    print("\n========================================")
    print("PTQ++ v3 Pipeline Complete")
    print("========================================\n")

    return model
