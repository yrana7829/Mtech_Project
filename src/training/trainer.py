import torch
from tqdm import tqdm
import os


class Trainer:

    def __init__(self, model, train_loader, val_loader, device, lr=3e-4):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total

        return total_loss / len(self.train_loader), acc

    def validate(self):

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.val_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total

        return acc

    def train(self, epochs, save_path="results/checkpoints/model.pth"):

        best_val_acc = 0

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(epochs):

            train_loss, train_acc = self.train_epoch()

            val_acc = self.validate()

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(self.model.state_dict(), save_path)

                print("Best model saved")
