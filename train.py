%%writefile medmnist-pytorch-benchmarks/train.py
import os, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import medmnist
from medmnist import PathMNIST

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_loaders(img_size=64, batch_size=128, num_workers=2):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    train_ds = PathMNIST(split='train', transform=transform, download=True)
    val_ds   = PathMNIST(split='val',   transform=transform, download=True)
    test_ds  = PathMNIST(split='test',  transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_classes = len(train_ds.info['label'])
    return train_loader, val_loader, test_loader, num_classes

def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc  = correct / total
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                vloss += loss.item() * images.size(0)
                _, pred = outputs.max(1)
                vtotal += labels.size(0)
                vcorrect += pred.eq(labels).sum().item()
        val_loss = vloss / vtotal
        val_acc  = vcorrect / vtotal

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    return train_losses, val_losses, val_accs

def evaluate(model, test_loader, device, num_classes, out_dir):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception as e:
        print("ROC-AUC calculation issue:", e)
        auc = float("nan")

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    return acc, auc

def save_curves(train_losses, val_losses, val_accs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    axes[0].plot(range(1, len(val_losses)+1),   val_losses,   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(range(1, len(val_accs)+1), val_accs, label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Validation Accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="PathMNIST ResNet18 baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, test_loader, num_classes = get_loaders(
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = build_model(num_classes).to(device)

    train_losses, val_losses, val_accs = train(model, train_loader, val_loader, device,
                                               epochs=args.epochs, lr=args.lr)

    # Save curves
    save_curves(train_losses, val_losses, val_accs, os.path.join(args.out_dir, "loss_acc_curves.png"))

    # Evaluate & save metrics
    test_acc, test_auc = evaluate(model, test_loader, device, num_classes, args.out_dir)
    print(f"Test Accuracy: {test_acc:.4f} | Test ROC-AUC (macro OVR): {test_auc:.4f}")

    metrics = {
        "test_accuracy": float(test_acc),
        "test_roc_auc_ovr_macro": float(test_auc),
        "val_acc_last": float(val_accs[-1]),
        "val_acc_best": float(max(val_accs))
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved:", os.path.join(args.out_dir, "metrics.json"))
    print("Saved:", os.path.join(args.out_dir, "loss_acc_curves.png"))
    print("Saved:", os.path.join(args.out_dir, "confusion_matrix.png"))

if __name__ == "__main__":
    main()

