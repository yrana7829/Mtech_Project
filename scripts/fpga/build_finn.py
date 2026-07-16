from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
)

# MODEL = "/home/yrana/project/models/mnv2_eurosat_ptqpp_v3_finn_int8.onnx"
MODEL = "/home/yrana/project/models/mnv2_eurosat_ptqpp_v3_finn_int8_o17.onnx"

cfg = DataflowBuildConfig(
    output_dir="/home/yrana/project/finn_build",
    synth_clk_period_ns=10.0,
    generate_outputs=[
        DataflowOutputType.ESTIMATE_REPORTS,
        DataflowOutputType.STITCHED_IP,
    ],
)

print("=" * 60)
print("Starting FINN Build")
print("=" * 60)

build_dataflow_cfg(MODEL, cfg)

print("\nFINN Build Complete")
