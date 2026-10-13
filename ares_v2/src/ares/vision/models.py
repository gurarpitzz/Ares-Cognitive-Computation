"""Neural Network Architectures for Visual Deepfake Detection.

Wraps timm vision backbones:
1. ResNet-50: Declared ARES visual backbone (Section X, Table XI).
2. Xception / Legacy Xception: Canonical FaceForensics++ benchmark baseline and Table XI CNN baseline.
3. EfficientNet-B4: Modern reference deepfake detector.
"""

from typing import Optional
import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    """Deepfake detector backbone with classification and feature extraction capabilities."""

    ARCH_MAP = {
        "resnet50": "resnet50.a1_in1k",
        "xception": "legacy_xception",
        "legacy_xception": "legacy_xception",
        "efficientnet_b4": "efficientnet_b4.ra2_in1k",
    }

    def __init__(
        self,
        architecture: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        arch_key = self.ARCH_MAP.get(architecture.lower(), architecture)
        
        # Instantiate backbone with pooled features
        self.backbone = timm.create_model(
            arch_key,
            pretrained=pretrained,
            num_classes=0,  # Strips default classification head; outputs pooled features
        )
        self.num_features = self.backbone.num_features
        
        # Binary classification head: Dropout -> Linear(num_features, 1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global average-pooled feature vectors [B, num_features]."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute frame-level logits [B, 1]."""
        features = self.extract_features(x)
        logits = self.head(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute frame-level deepfake probabilities [B, 1]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_parameter_count(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
