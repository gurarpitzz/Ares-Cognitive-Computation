"""Unit Tests for Audio Module (Milestone M08).

Verifies:
1. Mathematical audit of Table XI audio rows (Audit Issue I-14).
2. Audio feature extractors (80-channel log-mel, LFCC).
3. Audio model architectures (DeclaredAudioCNN ~8M params, LFCCLCNN, TraditionalAudioML).
4. Speaker embedding verification and EER calculation (Task B).
5. Silence shortcut audit (RF1).
6. Audio-Visual lip-sync offset calculation (Task C, Eq. 9).
"""

import numpy as np
import pytest
import torch

from ares.audio.audit_table11 import (
    calculate_precision,
    calculate_f1,
    solve_class_ratio_for_f1,
    audit_table11_audio_rows,
)
from ares.audio.features import (
    LogMelSpectrogramExtractor,
    LFCCExtractor,
    pad_crop_waveform,
    measure_silence_durations,
    trim_silence,
)
from ares.audio.models import DeclaredAudioCNN, LFCCLCNN, TraditionalAudioML
from ares.audio.speaker import compute_eer, compute_cosine_similarity
from ares.audio.silence import SilenceShortcutAuditor
from ares.audio.av_sync import extract_audio_envelope, compute_av_offset_sync


def test_audit_table11_audio_mathematical_proof():
    """Verify that Table XI audio rows are mathematically impossible under balanced classes."""
    # Row 2: ARES (Voice Embedding): TPR=0.97, FPR=0.03, Reported F1=0.98
    tpr_voice = 0.97
    fpr_voice = 0.03
    f1_voice_bal = calculate_f1(tpr_voice, fpr_voice, r=1.0)
    assert np.isclose(f1_voice_bal, 0.9700), f"Expected 0.9700, got {f1_voice_bal}"
    r_voice = solve_class_ratio_for_f1(tpr_voice, fpr_voice, 0.98)
    assert np.isclose(r_voice, 0.31972789), f"Unexpected solved ratio: {r_voice}"

    # Row 4: Traditional Audio ML: TPR=0.83, FPR=0.12, Reported F1=0.88
    tpr_trad = 0.83
    fpr_trad = 0.12
    f1_trad_bal = calculate_f1(tpr_trad, fpr_trad, r=1.0)
    assert np.isclose(f1_trad_bal, 0.85128205), f"Expected 0.8513, got {f1_trad_bal}"
    r_trad = solve_class_ratio_for_f1(tpr_trad, fpr_trad, 0.88)
    assert np.isclose(r_trad, 0.46969696), f"Unexpected solved ratio: {r_trad}"

    df = audit_table11_audio_rows()
    assert len(df) == 2
    assert "ARES (Voice Embedding)" in df["model"].values
    assert "Traditional Audio ML" in df["model"].values


def test_audio_feature_extractors():
    """Verify 80-channel log-mel and LFCC feature pipelines."""
    wav = torch.randn(2, 64000)  # 4 seconds at 16 kHz
    
    # 80-channel log-mel (Appendix A-c)
    mel_ext = LogMelSpectrogramExtractor(n_mels=80)
    mel = mel_ext(wav)
    assert mel.shape[0] == 2
    assert mel.shape[1] == 80
    assert mel.shape[2] > 350
    assert not torch.isnan(mel).any()

    # LFCC
    lfcc_ext = LFCCExtractor(n_lfcc=40)
    lfcc = lfcc_ext(wav)
    assert lfcc.shape[0] == 2
    assert lfcc.shape[1] == 40
    assert not torch.isnan(lfcc).any()

    # Padding and cropping
    short_wav = torch.randn(1, 16000)
    padded = pad_crop_waveform(short_wav, target_samples=64000)
    assert padded.shape == (1, 64000)
    long_wav = torch.randn(1, 100000)
    cropped = pad_crop_waveform(long_wav, target_samples=64000)
    assert cropped.shape == (1, 64000)


def test_audio_models_architectures():
    """Verify DeclaredAudioCNN (~8M params), LFCCLCNN, and TraditionalAudioML."""
    # DeclaredAudioCNN
    model = DeclaredAudioCNN()
    param_count = sum(p.numel() for p in model.parameters())
    # Should be ~8M params (between 7M and 10M)
    assert 7_000_000 <= param_count <= 10_000_000, f"Param count {param_count} out of range"
    
    mel = torch.randn(2, 80, 401)
    logits, emb = model(mel)
    assert logits.shape == (2, 2)
    assert emb.shape == (2, 512)

    # LFCCLCNN
    lcnn = LFCCLCNN()
    lfcc = torch.randn(2, 40, 401)
    logits_l, emb_l = lcnn(lfcc)
    assert logits_l.shape == (2, 2)
    assert emb_l.shape == (2, 256)

    # TraditionalAudioML
    trad = TraditionalAudioML()
    feat = TraditionalAudioML.extract_features(lfcc.numpy())
    assert feat.shape == (2, 160)
    X = np.random.randn(20, 160)
    y = np.array([0] * 10 + [1] * 10)
    trad.fit(X, y)
    probs = trad.predict_proba(X)
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_speaker_similarity_and_eer():
    """Verify EER calculation and cosine similarity."""
    bonafide = np.array([0.9, 0.85, 0.88, 0.92, 0.95, 0.80])
    spoof = np.array([0.1, 0.15, 0.20, 0.25, 0.05, 0.30])
    eer, thresh = compute_eer(bonafide, spoof)
    assert eer == 0.0
    assert 0.30 <= thresh <= 0.80

    # Overlapping distributions
    b_overlap = np.array([0.8, 0.7, 0.6, 0.5])
    s_overlap = np.array([0.7, 0.6, 0.5, 0.4])
    eer_ov, _ = compute_eer(b_overlap, s_overlap)
    assert 0.0 < eer_ov < 1.0

    # Cosine similarity
    v1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v2 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sim = compute_cosine_similarity(v1, v2)
    assert torch.isclose(sim[0], torch.tensor(1.0))
    assert torch.isclose(sim[1], torch.tensor(0.0))


def test_silence_shortcut_audit():
    """Verify silence duration measurement, trimming, and auditor (RF1)."""
    # Create synthetic waveform with known silence
    fs = 16000
    leading_silence = torch.zeros(1, int(fs * 0.5))  # 0.5s silence
    speech = torch.randn(1, int(fs * 1.0))          # 1.0s speech
    trailing_silence = torch.zeros(1, int(fs * 0.3)) # 0.3s silence
    wav = torch.cat([leading_silence, speech, trailing_silence], dim=1)

    stats = measure_silence_durations(wav, sample_rate=fs)
    assert 0.4 <= stats["leading_silence_sec"] <= 0.6
    assert 0.2 <= stats["trailing_silence_sec"] <= 0.4

    trimmed = trim_silence(wav, sample_rate=fs)
    assert len(trimmed.squeeze()) < len(wav.squeeze())

    # Auditor fitting
    auditor = SilenceShortcutAuditor()
    X_silence = np.random.randn(30, 5)
    y_labels = np.array([0] * 15 + [1] * 15)
    auditor.fit_silence_baseline(X_silence, y_labels)
    eval_res = auditor.evaluate_silence_baseline(X_silence, y_labels)
    assert "silence_only_auc" in eval_res
    assert "silence_only_eer" in eval_res


def test_av_sync_and_offset():
    """Verify audio envelope extraction and lip-sync offset calculation (Eq. 9)."""
    fs = 16000
    fps = 25.0
    t = np.linspace(0, 2.0, int(fs * 2.0))
    # Synthetic speech modulated carrier
    carrier = np.sin(2 * np.pi * 400 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))  # 3 Hz rhythm
    wav = carrier * modulator

    env = extract_audio_envelope(wav, sample_rate=fs, target_fps=fps)
    assert len(env) == int(2.0 * fps)
    assert np.max(env) > 0.0

    # In-sync visual signal
    lip_sync = env.copy()
    offset_0, conf_0 = compute_av_offset_sync(env, lip_sync, fps=fps)
    assert np.isclose(offset_0, 0.0, atol=0.05)
    assert conf_0 > 0.90

    # Out-of-sync visual signal (shifted by 5 frames = 200 ms)
    lip_shifted = np.roll(env, 5)
    offset_5, conf_5 = compute_av_offset_sync(env, lip_shifted, fps=fps)
    assert np.isclose(offset_5, 5 / fps, atol=0.05)
