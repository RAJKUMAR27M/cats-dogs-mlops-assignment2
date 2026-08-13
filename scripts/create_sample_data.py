"""
Generate a minimal sample dataset (30 cats + 30 dogs) for evaluators
who don't have Kaggle credentials.

Images are synthetic but visually distinguishable:
  Cats  – warm orange-tinted circles on light background
  Dogs  – cool brown-tinted rectangles on darker background

Usage:
    python scripts/create_sample_data.py        # creates sample_data/
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_PER_CLASS = 30
IMG_SIZE = 128          # kept small so zip stays tiny; resized to 224 during training
OUT_DIR = Path("sample_data")


def _cat_image(seed: int) -> Image.Image:
    """Synthetic cat: orange circle + pointy ears on pale background."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE),
                    tuple(rng.integers(220, 240, 3).tolist()))
    draw = ImageDraw.Draw(img)
    cx, cy, r = IMG_SIZE // 2, IMG_SIZE // 2 + 10, IMG_SIZE // 3
    # body circle
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=tuple(rng.integers(200, 240, size=3).tolist()))
    # ears (triangles)
    ear_col = tuple(rng.integers(180, 220, size=3).tolist())
    draw.polygon([(cx - r, cy - r), (cx - r - 15, cy - r - 25), (cx - r + 15, cy - r - 5)],
                 fill=ear_col)
    draw.polygon([(cx + r, cy - r), (cx + r - 15, cy - r - 5), (cx + r + 15, cy - r - 25)],
                 fill=ear_col)
    # eyes
    for ex in [cx - r // 3, cx + r // 3]:
        draw.ellipse([ex - 5, cy - 8, ex + 5, cy + 2],
                     fill=tuple(rng.integers(30, 100, size=3).tolist()))
    return img


def _dog_image(seed: int) -> Image.Image:
    """Synthetic dog: brown rectangle body + floppy ears on grey background."""
    rng = np.random.default_rng(seed + 1000)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE),
                    tuple(rng.integers(100, 140, size=3).tolist()))
    draw = ImageDraw.Draw(img)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    bw, bh = IMG_SIZE // 2, IMG_SIZE // 3
    body_col = tuple(rng.integers(120, 180, size=3).tolist())
    # body rectangle
    draw.rectangle([cx - bw, cy - bh, cx + bw, cy + bh], fill=body_col)
    # floppy ears
    ear_col = tuple(rng.integers(90, 140, size=3).tolist())
    draw.ellipse([cx - bw - 18, cy - bh, cx - bw + 10, cy + 10], fill=ear_col)
    draw.ellipse([cx + bw - 10, cy - bh, cx + bw + 18, cy + 10], fill=ear_col)
    # snout
    draw.ellipse([cx - 15, cy, cx + 15, cy + 20],
                 fill=tuple(rng.integers(160, 210, size=3).tolist()))
    # eyes
    for ex in [cx - bw // 3, cx + bw // 3]:
        draw.ellipse([ex - 5, cy - bh // 2, ex + 5, cy - bh // 2 + 10],
                     fill=(30, 20, 10))
    return img


def create_sample_data(out_dir: Path = OUT_DIR, n: int = N_PER_CLASS) -> None:
    for cls in ("Cat", "Dog"):
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    for i in range(n):
        _cat_image(i).save(str(out_dir / "Cat" / f"cat_{i:03d}.jpg"),
                           quality=75, optimize=True)
        _dog_image(i).save(str(out_dir / "Dog" / f"dog_{i:03d}.jpg"),
                           quality=75, optimize=True)

    cat_count = len(list((out_dir / "Cat").glob("*.jpg")))
    dog_count = len(list((out_dir / "Dog").glob("*.jpg")))
    size_kb = sum(f.stat().st_size for f in out_dir.rglob("*.jpg")) // 1024
    logger.info("Created %d cat + %d dog sample images in %s  (%d KB total)",
                cat_count, dog_count, out_dir, size_kb)


if __name__ == "__main__":
    create_sample_data()
