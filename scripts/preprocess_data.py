"""DVC stage helper: organise raw images into train/val/test splits."""

import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import organize_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",       default=None,
                        help="Override raw data directory (default: value in params.yaml)")
    parser.add_argument("--processed-dir", default=None,
                        help="Override processed data directory")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    data = params.get("data", {})
    raw_dir       = args.raw_dir       or data.get("raw_dir",       "data/raw")
    processed_dir = args.processed_dir or data.get("processed_dir", "data/processed")
    train_split   = params.get("train_split", 0.8)
    val_split     = params.get("val_split",   0.1)

    organize_dataset(raw_dir, processed_dir, train_split=train_split, val_split=val_split)
    print(f"Preprocessing complete → {processed_dir}")

