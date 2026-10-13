"""Forensic Auxiliary Feature Extraction & Late Fusion (Section X-B).

Operationalizes:
1. Azimuthally averaged FFT/DCT power spectrum features per face crop (Section X-B item 1).
2. Facial landmark temporal geometry: eye aspect ratio (EAR) jitter, mouth opening dynamics,
   and inter-frame coordinate jitter (Section X-B item 3).
3. Forensic classifier and late fusion with CNN scores.
"""

from typing import Optional, Union
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def compute_azimuthal_average_fft(image: Union[np.ndarray, Image.Image], num_bins: int = 32) -> np.ndarray:
    """Compute azimuthally averaged 1D radial profile of the 2D FFT power spectrum.
    
    Captures high-frequency GAN / synthesis artifacts and spectral decay characteristics.
    """
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            # RGB to Gray using standard luminance weights
            gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        elif image.ndim == 3 and image.shape[0] == 3:
            # CHW to HWC
            gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        else:
            gray = image.squeeze().astype(np.float32)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # 2D FFT and shift zero-frequency component to center
    fft2 = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft2)
    psd2D = np.abs(fft_shifted) ** 2

    # Coordinate grids centered at image midpoint
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - center_x, y - center_y)

    # Bin radial frequencies
    max_r = min(center_y, center_x)
    bin_edges = np.linspace(0, max_r, num_bins + 1)
    
    radial_profile = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            radial_profile[i] = np.mean(psd2D[mask])
        else:
            radial_profile[i] = 0.0

    # Log-transform power spectrum to stabilize variance
    return np.log1p(radial_profile)


def compute_eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio (EAR) for 68-point facial landmarks.
    
    Left eye: landmarks 36-41 (0-indexed)
    Right eye: landmarks 42-47 (0-indexed)
    EAR = (|p1 - p5| + |p2 - p4|) / (2 * |p0 - p3|)
    """
    if landmarks.shape[0] < 48:
        return 0.25  # Nominal neutral EAR
        
    def _ear(pts):
        vertical_1 = np.linalg.norm(pts[1] - pts[5])
        vertical_2 = np.linalg.norm(pts[2] - pts[4])
        horizontal = np.linalg.norm(pts[0] - pts[3])
        return (vertical_1 + vertical_2) / max(1e-6, 2.0 * horizontal)

    left_ear = _ear(landmarks[36:42])
    right_ear = _ear(landmarks[42:48])
    return float(0.5 * (left_ear + right_ear))


def compute_mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio (MAR) for 68-point facial landmarks.
    
    Inner mouth points: 61, 67 and 63, 65 (vertical), 60, 64 (horizontal)
    """
    if landmarks.shape[0] < 68:
        return 0.15  # Nominal neutral MAR
        
    p61, p67 = landmarks[61], landmarks[67]
    p63, p65 = landmarks[63], landmarks[65]
    p60, p64 = landmarks[60], landmarks[64]
    
    vertical = 0.5 * (np.linalg.norm(p61 - p67) + np.linalg.norm(p63 - p65))
    horizontal = np.linalg.norm(p60 - p64)
    return float(vertical / max(1e-6, horizontal))


def extract_video_landmark_jitter(landmarks_sequence: list[np.ndarray]) -> dict[str, float]:
    """Extract temporal geometry dynamics and inter-frame landmark jitter.
    
    Features extracted:
    - ear_variance: Temporal variance of Eye Aspect Ratio.
    - ear_mean: Mean EAR.
    - mar_variance: Temporal variance of Mouth Aspect Ratio.
    - landmark_jitter_mean: Mean inter-frame euclidean coordinate displacement.
    - landmark_jitter_std: Standard deviation of inter-frame jitter.
    - landmark_jitter_max: Peak displacement.
    """
    if not landmarks_sequence or len(landmarks_sequence) < 2:
        return {
            "ear_variance": 0.0,
            "ear_mean": 0.25,
            "mar_variance": 0.0,
            "landmark_jitter_mean": 0.0,
            "landmark_jitter_std": 0.0,
            "landmark_jitter_max": 0.0,
        }

    ears = [compute_eye_aspect_ratio(lm) for lm in landmarks_sequence]
    mars = [compute_mouth_aspect_ratio(lm) for lm in landmarks_sequence]
    
    jitters = []
    for t in range(1, len(landmarks_sequence)):
        curr_lm = landmarks_sequence[t]
        prev_lm = landmarks_sequence[t - 1]
        
        # Inter-ocular distance normalization (landmarks 36 and 45)
        if curr_lm.shape[0] >= 46:
            iod = np.linalg.norm(curr_lm[36] - curr_lm[45])
        else:
            iod = 100.0
        iod = max(10.0, iod)
        
        disp = np.linalg.norm(curr_lm - prev_lm, axis=1) / iod
        jitters.append(float(np.mean(disp)))

    return {
        "ear_variance": float(np.var(ears)),
        "ear_mean": float(np.mean(ears)),
        "mar_variance": float(np.var(mars)),
        "landmark_jitter_mean": float(np.mean(jitters)) if jitters else 0.0,
        "landmark_jitter_std": float(np.std(jitters)) if jitters else 0.0,
        "landmark_jitter_max": float(np.max(jitters)) if jitters else 0.0,
    }


class ForensicFeatureClassifier:
    """Classifier operating on combined frequency and landmark temporal features."""

    def __init__(self, classifier_type: str = "gbm", seed: int = 42) -> None:
        self.classifier_type = classifier_type
        self.seed = seed
        
        if classifier_type == "logistic":
            base_clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=seed,
            )
        elif classifier_type == "gbm":
            base_clf = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", base_clf),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForensicFeatureClassifier":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


class LateFusionDetector:
    """Late fusion head combining CNN video probability with forensic auxiliary score."""

    def __init__(self, alpha: float = 0.75) -> None:
        """alpha: Weight given to CNN score (1 - alpha given to forensic score)."""
        self.alpha = alpha

    def fuse(self, cnn_scores: np.ndarray, forensic_scores: np.ndarray) -> np.ndarray:
        """Compute fused probability score."""
        return self.alpha * cnn_scores + (1.0 - self.alpha) * forensic_scores

    def optimize_alpha(
        self,
        cnn_scores_val: np.ndarray,
        forensic_scores_val: np.ndarray,
        y_val: np.ndarray,
        grid_steps: int = 21,
    ) -> float:
        """Find optimal alpha on validation set to maximize ROC-AUC."""
        from sklearn.metrics import roc_auc_score
        
        best_alpha = 0.75
        best_auc = -1.0
        
        for candidate_alpha in np.linspace(0.0, 1.0, grid_steps):
            fused = candidate_alpha * cnn_scores_val + (1.0 - candidate_alpha) * forensic_scores_val
            try:
                score = roc_auc_score(y_val, fused)
            except Exception:
                score = 0.5
            if score > best_auc:
                best_auc = score
                best_alpha = float(candidate_alpha)
                
        self.alpha = best_alpha
        return self.alpha
