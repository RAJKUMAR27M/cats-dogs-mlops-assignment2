"""Training script for Cats vs Dogs CNN model with MLflow experiment tracking (PyTorch)."""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import classification_report, confusion_matrix

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import create_data_generators, organize_dataset
from src.models.cnn_model import build_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_curves(train_acc, val_acc, train_loss, val_loss, save_path: str) -> None:
    epochs = range(1, len(train_acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_acc, "b-o", ms=4, lw=2, label="Train")
    axes[0].plot(epochs, val_acc,   "r-s", ms=4, lw=2, label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, train_loss, "b-o", ms=4, lw=2, label="Train")
    axes[1].plot(epochs, val_loss,   "r-s", ms=4, lw=2, label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves → %s", save_path)


def _plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    try:
        import seaborn as sns
    except ImportError:
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix → %s", save_path)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, optimizer, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += ((outputs > 0.5).long() == labels.long()).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def load_params(params_file: str = "params.yaml") -> dict:
    p = PROJECT_ROOT / params_file
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _create_synthetic_data(tmp_dir: str, n_per_class: int = 10) -> None:
    from PIL import Image as PILImage
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            d = Path(tmp_dir) / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                PILImage.fromarray(arr).save(str(d / f"img_{i}.jpg"))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(data_dir: str, artifacts_dir: str, params: dict) -> dict:
    artifacts_path = Path(artifacts_dir)
    (artifacts_path / "model").mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(params.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(params.get("experiment_name", "cats-dogs-classification"))

    with mlflow.start_run(run_name=f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            "framework": "pytorch", "device": str(DEVICE),
            "batch_size": params.get("batch_size", 32),
            "epochs": params.get("epochs", 20),
            "learning_rate": params.get("learning_rate", 0.001),
            "dropout_rate": params.get("dropout_rate", 0.25),
        })

        train_loader, val_loader, test_loader = create_data_generators(
            data_dir, batch_size=params.get("batch_size", 32))

        model = build_model(dropout_rate=params.get("dropout_rate", 0.25)).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params.get("learning_rate", 0.001))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=3)

        best_val_acc, patience_count = 0.0, 0
        patience = params.get("patience", 5)
        epochs = params.get("epochs", 20)

        hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
        best_state = None

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, True)
            vl_loss, vl_acc = _run_epoch(model, val_loader,   criterion, optimizer, False)
            scheduler.step(vl_loss)

            hist["train_acc"].append(tr_acc); hist["val_acc"].append(vl_acc)
            hist["train_loss"].append(tr_loss); hist["val_loss"].append(vl_loss)

            mlflow.log_metrics({"train_acc": tr_acc, "val_acc": vl_acc,
                                "train_loss": tr_loss, "val_loss": vl_loss}, step=epoch)
            logger.info("Epoch %d/%d  train_acc=%.4f  val_acc=%.4f  val_loss=%.4f",
                        epoch, epochs, tr_acc, vl_acc, vl_loss)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate on test set
        _, test_acc = _run_epoch(model, test_loader, criterion, optimizer, False)
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                preds = (model(images.to(DEVICE)) > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        y_pred, y_true = np.array(all_preds), np.array(all_labels)
        mlflow.log_metric("test_accuracy", test_acc)
        logger.info("Test accuracy: %.4f", test_acc)

        # Save artifacts
        curves_path = str(artifacts_path / "training_curves.png")
        cm_path     = str(artifacts_path / "confusion_matrix.png")
        _plot_curves(hist["train_acc"], hist["val_acc"],
                     hist["train_loss"], hist["val_loss"], curves_path)
        _plot_confusion_matrix(y_true, y_pred, cm_path)

        model_path = str(artifacts_path / "model" / "model.pt")
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved → %s", model_path)

        report = classification_report(y_true, y_pred, target_names=["cat", "dog"])
        logger.info("\n%s", report)
        report_path = str(artifacts_path / "classification_report.txt")
        Path(report_path).write_text(report)

        for f in [curves_path, cm_path, report_path]:
            if Path(f).exists():
                mlflow.log_artifact(f)

        mlflow.pytorch.log_model(model, "model")

        final_metrics = {"test_accuracy": test_acc,
                         "train_accuracy": hist["train_acc"][-1],
                         "val_accuracy":   hist["val_acc"][-1]}
        metrics_path = str(artifacts_path / "metrics.json")
        Path(metrics_path).write_text(json.dumps(final_metrics, indent=2))
        mlflow.log_artifact(metrics_path)

        return final_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs CNN (PyTorch)")
    parser.add_argument("--data-dir",      default="data/processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--params",        default="params.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic data — no dataset required")
    args = parser.parse_args()

    params   = load_params(args.params)
    data_dir = args.data_dir

    if args.dry_run:
        logger.info("DRY-RUN mode: generating synthetic dataset")
        tmp_dir = tempfile.mkdtemp()
        _create_synthetic_data(tmp_dir)
        data_dir = tmp_dir
        params["epochs"] = 1
        params["batch_size"] = 4

    train(data_dir, args.artifacts_dir, params)


if __name__ == "__main__":
    main()
