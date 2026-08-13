"""Data preprocessing utilities for Cats vs Dogs classification."""

import os
import shutil
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_and_resize_image(image_path: str, target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load and resize an image to the target size (returns RGB numpy array)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    return np.array(img)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values from [0, 255] to [0, 1]."""
    return image.astype(np.float32) / 255.0


def preprocess_image(image_path: str, target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Load, resize, and normalize an image in one step."""
    img = load_and_resize_image(image_path, target_size)
    return normalize_image(img)


def organize_dataset(
    raw_dir: str,
    processed_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> None:
    """
    Organize raw dataset into train/val/test splits.

    Expected raw structure:  raw_dir/Cat/*.jpg  and  raw_dir/Dog/*.jpg
    Output structure:        processed_dir/{train,val,test}/{cat,dog}/*.jpg
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            (processed_path / split / cls).mkdir(parents=True, exist_ok=True)

    class_map = [("cat", "Cat"), ("dog", "Dog")]
    for class_name, folder_name in class_map:
        class_dir = raw_path / folder_name
        if not class_dir.exists():
            class_dir = raw_path / class_name
        if not class_dir.exists():
            logger.warning("Directory %s not found, skipping %s", class_dir, class_name)
            continue

        images = sorted(
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        np.random.seed(42)
        np.random.shuffle(images)

        n = len(images)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in splits.items():
            for img_path in split_images:
                dst = processed_path / split_name / class_name / img_path.name
                shutil.copy2(str(img_path), str(dst))

        logger.info(
            "%s: %d train, %d val, %d test",
            class_name,
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )


def create_data_generators(data_dir: str, batch_size: int = BATCH_SIZE):
    """Create train/val/test DataLoaders with augmentation on train set."""
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    data_path = Path(data_dir)
    train_ds = datasets.ImageFolder(str(data_path / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(data_path / "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(str(data_path / "test"),  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
