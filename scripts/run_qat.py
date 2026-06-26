import argparse
from xml.parsers.expat import model
import torch
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(project_root)

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate

from src.quantization.qat.qat_prepare import prepare_mobilenetv2_qat

from src.quantization.qat.qat_trainer import QATTrainer

from src.quantization.qat.qat_convert import convert_qat_model


def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")

    train_loader, val_loader, test_loader = get_dataset(args.dataset)

    print("Loading FP32 model...")

    model = get_model(args.model, num_classes=args.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict(checkpoint)
    fp32_acc = evaluate(model, test_loader, device)
    print("FP32 accuracy:", fp32_acc)

    # print("Preparing QAT model...")

    # qat_model = prepare_mobilenetv2_qat(model)
    # print(f"The prepared QAT model is : {qat_model}")
    # print("Evaluating QAT-prepared model...")

    # qat_model.eval()
    # qat_prepared_acc = evaluate(qat_model, test_loader, device)
    # print(f"QAT Prepared Accuracy = {qat_prepared_acc:.4f}")

    print("Preparing QAT model...")

    qat_model = prepare_mobilenetv2_qat(model)

    print("\n===== QAT DIAGNOSTICS =====")

    # --------------------------------------------------
    # EVAL MODE
    # --------------------------------------------------

    qat_model.eval()

    qat_val_acc = evaluate(qat_model, val_loader, device)

    qat_test_acc = evaluate(qat_model, test_loader, device)

    print(f"QAT Prepared VAL Accuracy = " f"{qat_val_acc:.4f}")

    print(f"QAT Prepared TEST Accuracy = " f"{qat_test_acc:.4f}")

    # --------------------------------------------------
    # TRAIN MODE
    # --------------------------------------------------

    qat_model.train()

    qat_train_mode_acc = evaluate(qat_model, test_loader, device)

    print(f"QAT Prepared TRAIN-MODE Accuracy = " f"{qat_train_mode_acc:.4f}")

    print("===== END DIAGNOSTICS =====\n")

    # IMPORTANT:
    # switch back to train mode before QAT training

    qat_model.train()

    trainer = QATTrainer(
        model=qat_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
    )

    print("Starting QAT fine-tuning...")

    trainer.train(epochs=args.epochs, save_path=args.qat_checkpoint)

    print("Loading best QAT model...")

    qat_model.load_state_dict(torch.load(args.qat_checkpoint, map_location=device))

    print("Converting to INT8...")

    quant_model = convert_qat_model(qat_model)

    print("Evaluating quantized model...")

    accuracy = evaluate(quant_model, test_loader, "cpu")

    print(f"QAT INT8 Test Accuracy: " f"{accuracy:.4f}")

    torch.save(quant_model.state_dict(), args.quant_save_path)

    print(f"Quantized model saved at " f"{args.quant_save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="eurosat")

    parser.add_argument("--model", type=str, default="mobilenetv2")

    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--qat_checkpoint",
        type=str,
        default=("results/checkpoints/" "mobilenetv2_eurosat_qat.pth"),
    )

    parser.add_argument(
        "--quant_save_path",
        type=str,
        default=("quantized_models/" "mobilenetv2_qat_int8.pth"),
    )

    args = parser.parse_args()

    main(args)
