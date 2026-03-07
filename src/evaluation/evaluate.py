import torch
from tqdm import tqdm


def evaluate(model, dataloader, device):

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy
