import torch
from src.dataset.dataloader import get_dataset
from src.evaluation.evaluate import evaluate

dataset = "eurosat"
model_name = "mobilenetv2"

device = torch.device("cpu")

print("Loading dataset...")
_, _, test_loader = get_dataset(dataset)

print("Loading PTQ model...")
model = torch.load(
    f"results/checkpoints/{dataset}_{model_name}_ptq.pth", map_location=device
)

model.eval()

print("Running evaluation...")
acc = evaluate(model, test_loader, device)

print("PTQ Accuracy:", acc * 100)
