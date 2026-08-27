import argparse, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from model import MammogramCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MammogramDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")
        image = self.transform(image)
        return image, int(row["label"])

def make_transforms(size=128):
    # Training augmentation follows the lecture examples.
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return train_tf, eval_tf

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if training:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    ys, preds, probs = [], [], []
    for images, labels in loader:
        out = model(images.to(DEVICE))
        p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        pred = (p >= 0.5).astype(int)
        ys.extend(labels.numpy())
        preds.extend(pred)
        probs.extend(p)
    return np.array(ys), np.array(preds), np.array(probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data_index.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="models/best_breast_cancer_cnn.pth")
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    data = pd.read_csv(args.index)
    missing = data.loc[~data["exists"]]
    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} images are missing. Run prepare_dataset.py with the correct --image-root."
        )

    train_df = data[data.split == "train"]
    val_df = data[data.split == "validation"]
    test_df = data[data.split == "test"]

    train_tf, eval_tf = make_transforms()
    train_ds = MammogramDataset(train_df, train_tf)
    val_ds = MammogramDataset(val_df, eval_tf)
    test_ds = MammogramDataset(test_df, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)

    # Class-weighted CE is used because the uploaded metadata are not perfectly balanced.
    counts = train_df["label"].value_counts().sort_index()
    weights = len(train_df) / (2 * torch.tensor(counts.values, dtype=torch.float32))
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    model = MammogramCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    print("Device:", DEVICE)
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "train_accuracy": train_acc, "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "class_names": ["benign", "malignant"],
                "image_size": 128
            }, args.out)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv("results/training_history.csv", index=False)

    model.load_state_dict(best_state)
    y, pred, prob = predict_all(model, test_loader)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")

    print("\nTEST RESULTS")
    print("Confusion matrix:\n", cm)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("ROC-AUC:", auc)
    print(classification_report(y, pred, target_names=["benign", "malignant"]))

    report = {
        "confusion_matrix": cm.tolist(),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
        "classification_report":
            classification_report(y, pred, target_names=["benign", "malignant"], output_dict=True)
    }
    Path("results/test_metrics.json").write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
