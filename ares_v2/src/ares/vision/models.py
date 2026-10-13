"""Neural Network Architectures for Visual Deepfake Detection.

Backbones:
1. ResNet-50: Declared ARES visual backbone (Section X, Table XI).
2. Xception / Legacy Xception: Canonical FaceForensics++ benchmark baseline and Table XI CNN baseline.
3. EfficientNet-B4: Modern reference deepfake detector.
"""

from typing import Dict
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm


class DeepfakeDetector(nn.Module):
    """Deepfake detector backbone with classification and feature extraction capabilities."""

    def __init__(
        self,
        architecture: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        
        if self.architecture in ["resnet50", "resnet-50"]:
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            m = tv_models.resnet50(weights=weights)
            self.num_features = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif self.architecture in ["efficientnet_b4", "efficientnet-b4"]:
            weights = tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            m = tv_models.efficientnet_b4(weights=weights)
            self.num_features = m.classifier[1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
        elif self.architecture in ["legacy_xception", "xception"]:
            self.backbone = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
            self.num_features = self.backbone.num_features
        else:
            self.backbone = timm.create_model(architecture, pretrained=pretrained, num_classes=0)
            self.num_features = self.backbone.num_features
        
        # Classification head: Dropout -> Linear(num_features, 1)
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

    def get_parameter_count(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
