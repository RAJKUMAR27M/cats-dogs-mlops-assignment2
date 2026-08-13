"""CNN model architecture for Cats vs Dogs binary classification (PyTorch)."""

import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    """4-block CNN with BatchNorm + Dropout; sigmoid output (0=cat, 1=dog)."""

    def __init__(self, dropout_rate: float = 0.25):
        super().__init__()

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate),
            )

        self.features = nn.Sequential(
            _block(3, 32),
            _block(32, 64),
            _block(64, 128),
            _block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def build_model(dropout_rate: float = 0.25) -> CatDogCNN:
    return CatDogCNN(dropout_rate=dropout_rate)
