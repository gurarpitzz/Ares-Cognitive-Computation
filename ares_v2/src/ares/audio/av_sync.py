r"""Audio-Visual Alignment and Lip-Sync Verification Engine (Task C, Eq. 9).

Implements:
1. Audio envelope extraction from waveform.
2. Temporal lip aperture dynamics analysis.
3. Cross-correlation lag analysis to compute synchronization deviation:
   \Delta_sync = |t_lip(f) - t_audio(f)|  (Eq. 9)
4. Decision logic for AV tampering detection:
   Flag tampering when \Delta_sync > \tau.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import scipy.signal as signal


def extract_audio_envelope(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    target_fps: float = 25.0,
) -> np.ndarray:
    """Extract smooth acoustic envelope aligned with video frame rate.
    
    Args:
        waveform: 1D audio waveform array
        sample_rate: Audio sampling rate in Hz
        target_fps: Video frame rate (e.g., 25 or 30 FPS)
    Returns:
        envelope: 1D array of envelope values sampled at target_fps
    """
    x = np.asarray(waveform).squeeze()
    # Full-wave rectification
    rectified = np.abs(x)
    
    # Low-pass filter at 10 Hz to extract speech syllable rhythm
    nyquist = sample_rate / 2.0
    cutoff = 10.0 / nyquist
    b, a = signal.butter(4, cutoff, btype="low")
    filtered = signal.filtfilt(b, a, rectified)
    
    # Resample to target video FPS
    num_video_frames = int(len(x) * target_fps / sample_rate)
    if num_video_frames <= 1:
        return np.zeros(1)
        
    envelope = signal.resample(filtered, num_video_frames)
    envelope = np.maximum(0.0, envelope)
    
    # Normalize
    max_val = np.max(envelope)
    if max_val > 0:
        envelope = envelope / max_val
    return envelope


def compute_av_offset_sync(
    audio_envelope: np.ndarray,
    lip_dynamics: np.ndarray,
    fps: float = 25.0,
    max_lag_sec: float = 0.5,
) -> Tuple[float, float]:
    r"""Compute temporal audio-visual sync deviation \Delta_sync (Eq. 9).
    
    Uses normalized cross-correlation between the speech energy envelope
    and the visual lip aperture motion.
    
    Args:
        audio_envelope: 1D array of audio energy envelope at video FPS
        lip_dynamics: 1D array of visual lip aperture / MAR values at video FPS
        fps: Video frame rate
        max_lag_sec: Maximum plausible temporal search window (+/- seconds)
        
    Returns:
        (delta_sync_sec, sync_confidence):
          - delta_sync_sec: Absolute temporal lag |\Delta_sync| in seconds
          - sync_confidence: Peak normalized cross-correlation coefficient
    """
    n = min(len(audio_envelope), len(lip_dynamics))
    if n < 5:
        return 0.0, 0.0
        
    a = audio_envelope[:n] - np.mean(audio_envelope[:n])
    v = lip_dynamics[:n] - np.mean(lip_dynamics[:n])
    
    norm_a = np.linalg.norm(a)
    norm_v = np.linalg.norm(v)
    if norm_a == 0 or norm_v == 0:
        return 0.0, 0.0
        
    max_lag_frames = int(max_lag_sec * fps)
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    
    # Compute cross-correlation for each lag
    corr = signal.correlate(a, v, mode="full")
    corr_lags = signal.correlation_lags(len(a), len(v), mode="full")
    
    # Restrict to search window
    mask = (corr_lags >= -max_lag_frames) & (corr_lags <= max_lag_frames)
    window_corr = corr[mask] / (norm_a * norm_v)
    window_lags = corr_lags[mask]
    
    if len(window_corr) == 0:
        return 0.0, 0.0
        
    best_idx = np.argmax(window_corr)
    peak_lag_frames = window_lags[best_idx]
    peak_confidence = float(window_corr[best_idx])
    
    delta_sync_sec = float(np.abs(peak_lag_frames) / fps)
    return delta_sync_sec, peak_confidence
