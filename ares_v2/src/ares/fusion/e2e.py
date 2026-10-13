"""End-to-End Multimodal Fusion Head (Appendix A + Eq. 14 Composite Loss).

Implements:
1. End-to-end multimodal projection architecture (Text + Vision -> Concat -> 1024-d MLP).
2. Composite loss function:
   L = lambda_cls * CE_weighted + lambda_adv * KL(f(x) || f(x + delta)) + lambda_reg * ||theta||^2
   with lambda = (1.0, 0.3, 1e-4), class weights (0.7, 1.3), eps = 0.02 (Claim C069).
3. Virtual Adversarial Training (VAT) perturbation in embedding space.
4. Training mechanism ablations: unweighted loss (C057) and lambda_adv = 0.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class MultimodalE2EFusionModel(nn.Module):
    """End-to-end multimodal fusion model for image-text (T3) and audio-visual (T6) tracks."""

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 512,
        fusion_dim: int = 1024,
        n_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 512)
        self.vision_proj = nn.Linear(vision_dim, 512)

        # Appendix-A 1024-d fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward_from_embeddings(self, e_text: torch.Tensor, e_vis: torch.Tensor) -> torch.Tensor:
        p_text = F.gelu(self.text_proj(e_text))
        p_vis = F.gelu(self.vision_proj(e_vis))
        joint = torch.cat([p_text, p_vis], dim=-1)
        logits = self.fusion_mlp(joint)
        return logits

    def forward(self, x_text: torch.Tensor, x_vis: torch.Tensor) -> torch.Tensor:
        return self.forward_from_embeddings(x_text, x_vis)


class E2ECompositeLoss(nn.Module):
    """Eq. 14 Composite Loss Function:
    L = lambda_cls * CE_weighted + lambda_adv * KL(f(x) || f(x + delta)) + lambda_reg * ||theta||^2
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_adv: float = 0.3,
        lambda_reg: float = 1e-4,
        eps: float = 0.02,
        xi: float = 1e-6,
        class_weights: Optional[Tuple[float, float]] = (0.7, 1.3),
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_adv = lambda_adv
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.xi = xi

        if class_weights is not None:
            weights_tensor = torch.tensor(list(class_weights), dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def compute_vat_perturbation(
        self,
        model: MultimodalE2EFusionModel,
        e_text: torch.Tensor,
        e_vis: torch.Tensor,
        logits_orig: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates Virtual Adversarial Training (VAT) perturbation delta in embedding space."""
        # Random initial perturbation
        d_text = torch.randn_like(e_text)
        d_vis = torch.randn_like(e_vis)

        # Normalize
        d_text = F.normalize(d_text, p=2, dim=-1)
        d_vis = F.normalize(d_vis, p=2, dim=-1)

        d_text.requires_grad_()
        d_vis.requires_grad_()

        # Forward perturbed
        perturbed_text = e_text + self.xi * d_text
        perturbed_vis = e_vis + self.xi * d_vis
        logits_pert = model.forward_from_embeddings(perturbed_text, perturbed_vis)

        # KL divergence KL(p_orig || p_pert)
        p_orig = F.softmax(logits_orig.detach(), dim=-1)
        log_p_pert = F.log_softmax(logits_pert, dim=-1)
        kl = F.kl_div(log_p_pert, p_orig, reduction="batchmean")

        # Gradients
        grad_text, grad_vis = torch.autograd.grad(kl, [d_text, d_vis])

        delta_text = self.eps * F.normalize(grad_text.detach(), p=2, dim=-1)
        delta_vis = self.eps * F.normalize(grad_vis.detach(), p=2, dim=-1)

        return delta_text, delta_vis

    def forward(
        self,
        model: MultimodalE2EFusionModel,
        e_text: torch.Tensor,
        e_vis: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = model.forward_from_embeddings(e_text, e_vis)

        # 1. Classification Loss (weighted CE)
        loss_cls = self.ce_loss(logits, targets)

        # 2. Adversarial Regularization via VAT
        if self.lambda_adv > 0:
            delta_text, delta_vis = self.compute_vat_perturbation(model, e_text, e_vis, logits)
            logits_adv = model.forward_from_embeddings(e_text + delta_text, e_vis + delta_vis)
            p_orig = F.softmax(logits.detach(), dim=-1)
            log_p_adv = F.log_softmax(logits_adv, dim=-1)
            loss_adv = F.kl_div(log_p_adv, p_orig, reduction="batchmean")
        else:
            loss_adv = torch.tensor(0.0, device=logits.device)

        # 3. Explicit L2 Parameter Regularization
        # Audit note: operates alongside AdamW weight decay (double regularization, Audit I-16)
        loss_reg = torch.tensor(0.0, device=logits.device)
        if self.lambda_reg > 0:
            for p in model.parameters():
                loss_reg = loss_reg + torch.sum(p ** 2)

        total_loss = (
            self.lambda_cls * loss_cls +
            self.lambda_adv * loss_adv +
            self.lambda_reg * loss_reg
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_adv": float(loss_adv.item()),
            "loss_reg": float(loss_reg.item()),
        }
        return total_loss, metrics


class E2ETrainer:
    """Trainer harness for the end-to-end multimodal fusion head."""

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 512,
        lambda_cls: float = 1.0,
        lambda_adv: float = 0.3,
        lambda_reg: float = 1e-4,
        eps: float = 0.02,
        class_weights: Optional[Tuple[float, float]] = (0.7, 1.3),
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 15,
        batch_size: int = 32,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        torch.manual_seed(seed)
        self.model = MultimodalE2EFusionModel(text_dim=text_dim, vision_dim=vision_dim).to(self.device)
        self.criterion = E2ECompositeLoss(
            lambda_cls=lambda_cls,
            lambda_adv=lambda_adv,
            lambda_reg=lambda_reg,
            eps=eps,
            class_weights=class_weights,
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.fitted = False

    def fit(
        self,
        X_text: np.ndarray,
        X_vis: np.ndarray,
        y: np.ndarray,
    ) -> "E2ETrainer":
        torch.manual_seed(self.seed)
        self.model.train()

        X_t_tensor = torch.tensor(X_text, dtype=torch.float32)
        X_v_tensor = torch.tensor(X_vis, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_t_tensor, X_v_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for b_text, b_vis, b_y in loader:
                b_text, b_vis, b_y = b_text.to(self.device), b_vis.to(self.device), b_y.to(self.device)
                self.optimizer.zero_grad()
                loss, _ = self.criterion(self.model, b_text, b_vis, b_y)
                loss.backward()
                self.optimizer.step()

        self.fitted = True
        return self

    def predict_proba(self, X_text: np.ndarray, X_vis: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        self.model.eval()
        with torch.no_grad():
            b_text = torch.tensor(X_text, dtype=torch.float32).to(self.device)
            b_vis = torch.tensor(X_vis, dtype=torch.float32).to(self.device)
            logits = self.model(b_text, b_vis)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X_text: np.ndarray, X_vis: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_text, X_vis)
        return np.argmax(probs, axis=1)
