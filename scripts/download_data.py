"""Download the Cats vs Dogs dataset from Kaggle."""

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATASET_ID = "bhavikjikadara/dog-and-cat-classification-dataset"
DEFAULT_RAW_DIR = "data/raw"


def download_dataset(output_dir: str = DEFAULT_RAW_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Accept credentials from env or ~/.kaggle/kaggle.json
    kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            logger.error("No Kaggle credentials found.")
            logger.error("Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place kaggle.json at ~/.kaggle/kaggle.json")
            sys.exit(1)

    try:
        import kaggle

        logger.info("Downloading %s …", DATASET_ID)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(DATASET_ID, path=str(output_path), unzip=True)
        logger.info("Dataset extracted to %s", output_dir)
        _verify(output_dir)
    except ImportError:
        logger.error("kaggle package missing – run: pip install kaggle")
        sys.exit(1)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)


def _verify(data_dir: str) -> bool:
    data_path = Path(data_dir)
    cats = data_path / "Cat"
    dogs = data_path / "Dog"

    if not cats.exists() or not dogs.exists():
        logger.warning("Expected Cat/ and Dog/ directories not found in %s", data_dir)
        logger.info("Contents: %s", [p.name for p in data_path.iterdir()])
        return False

    n_cats = len(list(cats.glob("*.jpg")))
    n_dogs = len(list(dogs.glob("*.jpg")))
    logger.info("Verified: %d cat images, %d dog images", n_cats, n_dogs)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Cats vs Dogs dataset from Kaggle")
    parser.add_argument("--output-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        sys.exit(0 if _verify(args.output_dir) else 1)

    download_dataset(args.output_dir)


if __name__ == "__main__":
    main()
