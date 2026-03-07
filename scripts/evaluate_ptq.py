import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset.dataloader import get_dataset
from src.models.model_loader import get_model
from src.evaluation.evaluate import evaluate


dataset = "eurosat"
model_name = "resnet18"

device = torch.device("cpu")

print("Loading dataset...")
_, _, test_loader = get_dataset(dataset)

print("Loading model...")
model = get_model(model_name, num_classes=10)

print("Loading PTQ checkpoint...")
model.load_state_dict(
    torch.load(
        f"results/checkpoints/{dataset}_{model_name}_ptq.pth", map_location=device
    )
)

model.eval()
model.to(device)

print("Running evaluation...")

acc = evaluate(model, test_loader, device)

print("PTQ Accuracy:", acc * 100)
