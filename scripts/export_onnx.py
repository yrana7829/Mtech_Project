import torch
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(project_root)


from src.models.model_loader import get_model


def export_model(args):

    device = torch.device("cpu")

    print("Loading model...")

    model = get_model(args.model, num_classes=args.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(checkpoint)

    model.eval()

    print("Creating dummy input...")

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    print("Exporting ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"ONNX model saved to: {args.output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()

    export_model(args)
