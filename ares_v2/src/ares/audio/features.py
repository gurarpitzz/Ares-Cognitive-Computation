"""Audio Feature Extraction and Preprocessing Pipeline.

Provides deterministic audio loading, resampling, fixed-length cropping/padding,
80-channel log-mel spectrogram extraction (per Appendix A-c), LFCC extraction,
and energy-based silence detection/trimming for silence shortcut control (RF1).
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T


TARGET_SAMPLE_RATE = 16000
DEFAULT_DURATION_SEC = 4.0
DEFAULT_SAMPLES = int(TARGET_SAMPLE_RATE * DEFAULT_DURATION_SEC)  # 64,000 samples


class LogMelSpectrogramExtractor:
    """Extracts 80-channel log-mel spectrograms as specified in Appendix A-c.
    
    Window length: 25 ms (400 samples at 16 kHz)
    Hop length: 10 ms (160 samples at 16 kHz)
    Number of mel filterbanks: 80
    """
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram from waveform.
        
        Args:
            waveform: Tensor of shape (..., time)
        Returns:
            log_mel: Tensor of shape (..., n_mels, time_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Ensure device consistency
        self.mel_transform = self.mel_transform.to(waveform.device)
        mel = self.mel_transform(waveform)
        # Log-mel with numerical stability
        log_mel = torch.log(torch.clamp(mel, min=1e-6))
        return log_mel


class LFCCExtractor:
    """Linear Frequency Cepstral Coefficients (LFCC) extractor.
    
    LFCCs use a linear filterbank in place of mel-scale warping, preserving
    high-frequency spectral cues characteristic of synthetic vocoder artifacts.
    """
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        n_filter: int = 40,
        n_lfcc: int = 40,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ):
        self.sample_rate = sample_rate
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
        )

    def _get_linear_filterbank(self, device: torch.device) -> torch.Tensor:
        n_freq = self.n_fft // 2 + 1
        filters = torch.zeros(self.n_filter, n_freq, device=device)
        delta_f = (self.sample_rate / 2.0) / (self.n_filter + 1)
        freq_bins = torch.linspace(0, self.sample_rate / 2.0, n_freq, device=device)
        
        for m in range(1, self.n_filter + 1):
            f_m_minus = (m - 1) * delta_f
            f_m = m * delta_f
            f_m_plus = (m + 1) * delta_f
            
            up_slope = (freq_bins - f_m_minus) / (f_m - f_m_minus + 1e-8)
            down_slope = (f_m_plus - freq_bins) / (f_m_plus - f_m + 1e-8)
            tri = torch.maximum(torch.zeros_like(up_slope), torch.minimum(up_slope, down_slope))
            filters[m - 1] = tri
        return filters

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract LFCC features from waveform.
        
        Args:
            waveform: Tensor of shape (..., time)
        Returns:
            lfcc: Tensor of shape (..., n_lfcc, time_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        self.spectrogram = self.spectrogram.to(waveform.device)
        spec = self.spectrogram(waveform)  # (batch, n_freq, frames)
        
        fb = self._get_linear_filterbank(waveform.device)  # (n_filter, n_freq)
        linear_energy = torch.matmul(fb, spec)  # (batch, n_filter, frames)
        log_energy = torch.log(torch.clamp(linear_energy, min=1e-6))
        
        # Apply Discrete Cosine Transform (DCT-II)
        # Using orthogonal DCT matrix
        n = torch.arange(self.n_filter, device=waveform.device).unsqueeze(1)
        k = torch.arange(self.n_lfcc, device=waveform.device).unsqueeze(0)
        dct_mat = torch.cos(np.pi / self.n_filter * (n + 0.5) * k) * np.sqrt(2.0 / self.n_filter)
        dct_mat[:, 0] *= 1.0 / np.sqrt(2.0)
        
        # (batch, frames, n_filter) x (n_filter, n_lfcc) -> (batch, frames, n_lfcc)
        lfcc = torch.matmul(log_energy.transpose(1, 2), dct_mat).transpose(1, 2)
        return lfcc


def pad_crop_waveform(
    waveform: torch.Tensor,
    target_samples: int = DEFAULT_SAMPLES,
) -> torch.Tensor:
    """Pad or crop a 1D or 2D audio waveform to exactly target_samples."""
    if waveform.dim() == 1:
        samples = waveform.size(0)
        if samples == target_samples:
            return waveform
        elif samples > target_samples:
            # Center crop or initial crop
            return waveform[:target_samples]
        else:
            # Repeat or zero-pad
            pad_amount = target_samples - samples
            return F.pad(waveform, (0, pad_amount))
    elif waveform.dim() == 2:
        samples = waveform.size(-1)
        if samples == target_samples:
            return waveform
        elif samples > target_samples:
            return waveform[:, :target_samples]
        else:
            pad_amount = target_samples - samples
            return F.pad(waveform, (0, pad_amount))
    else:
        raise ValueError(f"Unsupported waveform dimension: {waveform.dim()}")


def measure_silence_durations(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    energy_threshold_db: float = -40.0,
) -> Dict[str, float]:
    """Measure leading, trailing, and total silence duration for silence-shortcut audit (RF1).
    
    Args:
        waveform: 1D Tensor or numpy array of audio samples
        sample_rate: Sampling rate in Hz
        frame_ms: Analysis frame length in ms
        hop_ms: Analysis hop length in ms
        energy_threshold_db: Threshold below peak energy in dB to classify as silence
        
    Returns:
        Dict with keys:
          'leading_silence_sec': Duration of initial silence
          'trailing_silence_sec': Duration of trailing silence
          'total_silence_sec': Total silence duration
          'total_duration_sec': Utterance duration
          'speech_duration_sec': Active speech duration
    """
    if isinstance(waveform, torch.Tensor):
        x = waveform.detach().cpu().squeeze().numpy()
    else:
        x = np.asarray(waveform).squeeze()
        
    total_duration = len(x) / sample_rate
    frame_len = int(sample_rate * frame_ms / 1000.0)
    hop_len = int(sample_rate * hop_ms / 1000.0)
    
    if len(x) < frame_len:
        return {
            "leading_silence_sec": 0.0,
            "trailing_silence_sec": 0.0,
            "total_silence_sec": 0.0,
            "total_duration_sec": total_duration,
            "speech_duration_sec": total_duration,
        }
        
    # Compute frame short-time energy
    num_frames = (len(x) - frame_len) // hop_len + 1
    energies = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_len
        frame = x[start : start + frame_len]
        energies[i] = np.sum(frame ** 2) + 1e-12
        
    max_energy = np.max(energies)
    energies_db = 10.0 * np.log10(energies / (max_energy + 1e-12))
    
    is_speech = energies_db > energy_threshold_db
    if not np.any(is_speech):
        # All silence
        return {
            "leading_silence_sec": total_duration,
            "trailing_silence_sec": total_duration,
            "total_silence_sec": total_duration,
            "total_duration_sec": total_duration,
            "speech_duration_sec": 0.0,
        }
        
    speech_indices = np.where(is_speech)[0]
    first_speech_frame = speech_indices[0]
    last_speech_frame = speech_indices[-1]
    
    leading_silence_sec = (first_speech_frame * hop_len) / sample_rate
    trailing_silence_sec = ((num_frames - 1 - last_speech_frame) * hop_len) / sample_rate
    total_silence_sec = (np.sum(~is_speech) * hop_len) / sample_rate
    speech_duration_sec = total_duration - total_silence_sec
    
    return {
        "leading_silence_sec": float(leading_silence_sec),
        "trailing_silence_sec": float(trailing_silence_sec),
        "total_silence_sec": float(total_silence_sec),
        "total_duration_sec": float(total_duration),
        "speech_duration_sec": float(max(0.0, speech_duration_sec)),
    }


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    energy_threshold_db: float = -40.0,
) -> torch.Tensor:
    """Trim leading and trailing silence from waveform for silence-shortcut control."""
    was_tensor = isinstance(waveform, torch.Tensor)
    device = waveform.device if was_tensor else None
    
    if was_tensor:
        x = waveform.detach().cpu().squeeze().numpy()
    else:
        x = np.asarray(waveform).squeeze()
        
    frame_len = int(sample_rate * 0.02)  # 20 ms
    hop_len = int(sample_rate * 0.01)    # 10 ms
    
    if len(x) < frame_len:
        return waveform
        
    num_frames = (len(x) - frame_len) // hop_len + 1
    energies = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_len
        frame = x[start : start + frame_len]
        energies[i] = np.sum(frame ** 2) + 1e-12
        
    max_energy = np.max(energies)
    energies_db = 10.0 * np.log10(energies / (max_energy + 1e-12))
    is_speech = energies_db > energy_threshold_db
    
    if not np.any(is_speech):
        return waveform
        
    speech_indices = np.where(is_speech)[0]
    first_sample = speech_indices[0] * hop_len
    last_sample = min(len(x), (speech_indices[-1] + 1) * hop_len + frame_len)
    
    trimmed = x[first_sample:last_sample]
    if was_tensor:
        return torch.from_numpy(trimmed).float().to(device)
    return trimmed
