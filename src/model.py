"""
src/model.py

CNN architecture for wafer defect classification.

Design decisions (see wiki/modeling-decisions.md):
- Three conv blocks with increasing filter depth (32 → 64 → 128)
- BatchNorm after each conv for stable training
- MaxPool halves spatial dims each block: 64x64 → 32x32 → 16x16 → 8x8
- Dropout(0.5) before final classifier to regularize
- Single input channel (grayscale wafer map)
"""

import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    """
    Custom CNN for 9-class wafer defect classification.

    Input:  (B, 1, 64, 64) — single-channel, normalized wafer map
    Output: (B, num_classes) — raw logits (no softmax; use CrossEntropyLoss)
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 64, 64) -> (32, 32, 32)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
