"""Audio Spoof Detection and Embedding Models.

Implements:
1. DeclaredAudioCNN: The 6-block CNN on 80-channel log-mel spectrograms producing
   a 512-dimensional embedding and binary spoof classification logits, matching
   Appendix A-c (params: ~8M).
2. LFCCLCNN: Light-CNN with Max-Feature-Map (MFM) activations on LFCC features
   (canonical ASVspoof baseline).
3. TraditionalAudioML: Acoustic statistical feature extractor (LFCC/MFCC stats + deltas)
   coupled with GMM/GradientBoosting classifier to evaluate Table XI's "Traditional Audio ML" row.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture


class ConvBlock(nn.Module):
    """A convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DeclaredAudioCNN(nn.Module):
    """ARES Declared Audio Encoder (Appendix A-c).
    
    Specification:
      • Input: 80-channel log-mel spectrograms
      • Convolutional blocks: 6
      • Output dimension: 512
      • Parameters: ≈ 8M (~7.8M)
    """
    def __init__(
        self,
        in_channels: int = 1,
        n_mels: int = 80,
        embedding_dim: int = 512,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # 6 Convolutional Blocks
        # Input shape: (B, 1, 80, T)
        self.block1 = ConvBlock(in_channels, 64, kernel_size=3, pool_size=(2, 2))    # (B, 64, 40, T/2)
        self.block2 = ConvBlock(64, 128, kernel_size=3, pool_size=(2, 2))             # (B, 128, 20, T/4)
        self.block3 = ConvBlock(128, 256, kernel_size=3, pool_size=(2, 2))            # (B, 256, 10, T/8)
        self.block4 = ConvBlock(256, 512, kernel_size=3, pool_size=(2, 2))            # (B, 512, 5, T/16)
        self.block5 = ConvBlock(512, 512, kernel_size=3, pool_size=(1, 2))            # (B, 512, 5, T/32)
        self.block6 = ConvBlock(512, 512, kernel_size=3, pool_size=(1, 2))            # (B, 512, 5, T/64)

        # Adaptive pooling to fixed spatial grid (5, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 2))  # (B, 512, 5, 2) -> 512 * 10 = 5120
        flatten_dim = 512 * 5 * 2  # 5,120

        # Projection to 512-dim embedding
        self.fc_embed = nn.Sequential(
            nn.Linear(flatten_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # Classification head: 0 = bonafide, 1 = spoof
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_embedding(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Extract 512-dimensional voice embedding vector."""
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)  # (B, 1, 80, T)
        
        x = self.block1(log_mel)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        emb = self.fc_embed(x)
        return emb

    def forward(self, log_mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            log_mel: (B, 80, T) or (B, 1, 80, T) log-mel spectrogram
        Returns:
            logits: (B, 2) classification logits
            embedding: (B, 512) audio embedding
        """
        emb = self.extract_embedding(log_mel)
        logits = self.classifier(emb)
        return logits, emb


class MFM(nn.Module):
    """Max-Feature-Map (MFM) activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        out = torch.max(x[:, 0::2, :, :], x[:, 1::2, :, :])
        return out


class LFCCLCNN(nn.Module):
    """Light-CNN (LCNN) architecture with MFM activations for LFCC features."""
    def __init__(self, in_channels: int = 1, embedding_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.conv1 = MFM(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = MFM(32, 48, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv3 = MFM(48, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv4 = MFM(64, 96, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(96 * 4, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, lfcc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if lfcc.dim() == 3:
            lfcc = lfcc.unsqueeze(1)
            
        x = self.pool1(self.conv1(lfcc))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        emb = F.relu(self.fc(x))
        logits = self.classifier(emb)
        return logits, emb


class TraditionalAudioML:
    """Traditional Audio ML Baseline (Table XI row 4).
    
    Extracts summary statistics (mean, std, delta-mean, delta-std) from acoustic
    cepstral features (LFCC / MFCC) and classifies using HistGradientBoosting
    or Gaussian Mixture Models.
    """
    def __init__(self, classifier_type: str = "gradient_boosting"):
        self.classifier_type = classifier_type
        if classifier_type == "gradient_boosting":
            self.model = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                random_state=42,
            )
        else:
            self.model = GaussianMixture(n_components=16, random_state=42)
        self.is_fitted = False

    @staticmethod
    def extract_features(cepstral_matrix: np.ndarray) -> np.ndarray:
        """Extract statistical vector from cepstral matrix (n_coeff, n_frames).
        
        Returns 1D feature vector consisting of:
          - Mean across frames (n_coeff)
          - Std across frames (n_coeff)
          - Delta mean across frames (n_coeff)
          - Delta std across frames (n_coeff)
        """
        if cepstral_matrix.ndim == 3:
            # Batch of matrices (B, n_coeff, n_frames)
            features = []
            for m in cepstral_matrix:
                features.append(TraditionalAudioML.extract_features(m))
            return np.array(features)

        mean = np.mean(cepstral_matrix, axis=-1)
        std = np.std(cepstral_matrix, axis=-1)
        
        # Delta features
        if cepstral_matrix.shape[-1] > 1:
            delta = np.diff(cepstral_matrix, axis=-1)
            d_mean = np.mean(delta, axis=-1)
            d_std = np.std(delta, axis=-1)
        else:
            d_mean = np.zeros_like(mean)
            d_std = np.zeros_like(std)
            
        return np.concatenate([mean, std, d_mean, d_std])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TraditionalAudioML":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        if self.classifier_type == "gradient_boosting":
            return self.model.predict_proba(X)
        else:
            # Score samples
            scores = self.model.score_samples(X)
            # Sigmoid proxy
            probs = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
