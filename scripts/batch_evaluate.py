"""
Batch post-deployment evaluation – collects predictions on test images
(real or synthetic) and logs metrics to MLflow (M5).
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _collect_predictions(base_url: str, data_dir: str):
    import requests
    from PIL import Image as PILImage

    predictions, true_labels, latencies = [], [], []
    test_dir = Path(data_dir) / "test"

    if test_dir.exists():
        logger.info("Using real test data from %s", test_dir)
        for cls_name, label in [("cat", 0), ("dog", 1)]:
            for img_path in list((test_dir / cls_name).glob("*.jpg"))[:50]:
                with open(img_path, "rb") as f:
                    _send(base_url, f, img_path.name, label,
                          predictions, true_labels, latencies)
    else:
        logger.info("Test data not found – using synthetic images")
        for i in range(20):
            label = i % 2
            arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = PILImage.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            _send(base_url, buf, f"synth_{i}.jpg", label,
                  predictions, true_labels, latencies)

    return predictions, true_labels, latencies


def _send(base_url, file_obj, filename, true_label, predictions, true_labels, latencies):
    import requests

    try:
        t0 = time.time()
        r = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, file_obj, "image/jpeg")},
            timeout=30,
        )
        latency = time.time() - t0
        if r.status_code == 200:
            data = r.json()
            pred = 1 if data["prediction"] == "dog" else 0
            predictions.append(pred)
            true_labels.append(true_label)
            latencies.append(latency)
    except Exception as exc:
        logger.warning("Skipping %s: %s", filename, exc)


def evaluate(base_url: str, data_dir: str, output_dir: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    preds, labels, latencies = _collect_predictions(base_url, data_dir)

    if not preds:
        logger.error("No predictions collected – is the service running?")
        return {}

    metrics = {
        "num_samples": len(preds),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "avg_latency_ms": float(np.mean(latencies) * 1000),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_file = out_path / "post_deployment_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics → %s", metrics_file)

    for k, v in metrics.items():
        logger.info("  %-22s %.4f" if isinstance(v, float) else "  %-22s %s", k, v)

    try:
        import mlflow

        mlflow.set_experiment("post-deployment-evaluation")
        with mlflow.start_run(run_name="batch_eval"):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_artifact(str(metrics_file))
        logger.info("Metrics logged to MLflow")
    except Exception as exc:
        logger.warning("MLflow logging skipped: %s", exc)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for post-deployment tracking")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    result = evaluate(args.base_url, args.data_dir, args.output_dir)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
