"""Speaker Embedding Verification and Similarity Engine (Task B).

Implements:
1. ECAPA-TDNN speaker embedding extraction (using SpeechBrain pretrained on VoxCeleb2).
2. Cosine similarity scoring per Eq. 10:
   sim(v_o, v_s) = (v_o . v_s) / (||v_o|| * ||v_s||)
3. Equal Error Rate (EER) and minDCF computation.
4. VoxCeleb1-O trial list evaluation (RF4 sanity check).
5. ASVspoof ASV protocol evaluation (target vs spoof trial separation).
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import SpeakerRecognition


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the optimal EER threshold.
    
    In verification:
      - bonafide_scores are scores of target/bonafide trials (higher should be bonafide).
      - spoof_scores are scores of non-target/spoof trials (lower should be non-target/spoof).
      
    Returns:
      (eer, threshold): Equal Error Rate as a float in [0, 1] and operating threshold.
    """
    bonafide = np.asarray(bonafide_scores, dtype=np.float64)
    spoof = np.asarray(spoof_scores, dtype=np.float64)
    
    if len(bonafide) == 0 or len(spoof) == 0:
        return 0.5, 0.0

    # Sort all scores to evaluate thresholds
    all_scores = np.concatenate([bonafide, spoof])
    thresholds = np.sort(np.unique(all_scores))
    
    # Calculate False Rejection Rate (FRR) and False Acceptance Rate (FAR)
    # FRR = P(bonafide < threshold)
    # FAR = P(spoof >= threshold)
    n_bonafide = len(bonafide)
    n_spoof = len(spoof)

    frr = np.searchsorted(np.sort(bonafide), thresholds, side="right") / n_bonafide
    far = (n_spoof - np.searchsorted(np.sort(spoof), thresholds, side="right")) / n_spoof

    # Find intersection point where |FAR - FRR| is minimized
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    
    eer = (far[idx] + frr[idx]) / 2.0
    threshold = float(thresholds[idx])
    return float(eer), threshold


def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """Compute normalized cosine similarity per Eq. 10.
    
    sim(v_o, v_s) = (v_o . v_s) / (||v_o|| * ||v_s||)
    """
    v1_norm = F.normalize(emb1, p=2, dim=-1)
    v2_norm = F.normalize(emb2, p=2, dim=-1)
    return (v1_norm * v2_norm).sum(dim=-1)


class ECAPASpeakerVerifier:
    """ECAPA-TDNN Speaker Verification Engine using SpeechBrain."""
    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "cache/models/ecapa_tdnn",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        # SpeechBrain expects run_opts device as torch.device or string
        run_device = "cuda:0" if device.startswith("cuda") else "cpu"
        self.classifier = SpeakerRecognition.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": run_device},
        )

    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to 192-dimensional ECAPA-TDNN embedding vector.
        
        Args:
            waveform: Tensor of shape (time,) or (1, time) at 16 kHz
        Returns:
            embedding: Normalized Tensor of shape (192,)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        with torch.no_grad():
            emb = self.classifier.encode_batch(waveform.to(self.classifier.device))
            emb = emb.squeeze(1).squeeze(0)  # (192,)
            emb_norm = F.normalize(emb, p=2, dim=-1)
        return emb_norm

    def verify_pair(self, wav1: torch.Tensor, wav2: torch.Tensor) -> float:
        """Verify speaker identity between two waveforms using cosine similarity (Eq. 10)."""
        emb1 = self.encode_waveform(wav1)
        emb2 = self.encode_waveform(wav2)
        score = float(compute_cosine_similarity(emb1, emb2).item())
        return score
