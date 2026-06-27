import argparse

from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(args):

    print("Loading FP32 ONNX model...")
    print(args.input_model)

    print("\nQuantizing...")

    quantize_dynamic(
        model_input=args.input_model,
        model_output=args.output_model,
        weight_type=QuantType.QInt8,
    )

    print("\nINT8 model saved to:")
    print(args.output_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_model", required=True)

    parser.add_argument("--output_model", required=True)

    args = parser.parse_args()

    quantize_model(args)
