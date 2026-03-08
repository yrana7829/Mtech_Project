import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate


from src.quantization.qat.qat_train import run_qat_training

import torch
import argparse


def main(args):

    train_loader, val_loader, test_loader = get_dataset(args.dataset)
    num_classes = len(train_loader.dataset.classes)

    model = get_model(args.model, num_classes=num_classes)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)

    print("Evaluating FP32 model")
    evaluate(model, test_loader)

    print("Running QAT training")

    qat_model = run_qat_training(model, train_loader, val_loader, epochs=args.epochs)

    print("Evaluating QAT model")

    evaluate(qat_model, test_loader)

    save_path = f"checkpoints/{args.model}_qat_{args.dataset}.pth"

    torch.save(qat_model.state_dict(), save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    main(args)
