# ARES Rebuild — Laptop Setup & Hardware Environment

**Document ID:** `docs/laptop_setup.md`  
**Milestone:** M00  
**Machine:** HP OMEN Gaming Laptop 16-wd0xxx  
**Host Operating System:** Microsoft Windows 11 Home Single Language (64-bit, Build 26200)  
**Linux Subsystem:** WSL2 Ubuntu 26.04 LTS (Resolute Raccoon)  
**Date:** October 13, 2026  

---

## 1. Physical Hardware Specifications

| Component | Specification | Operational Role |
| :--- | :--- | :--- |
| **Processor (CPU)** | 13th Gen Intel® Core™ i7-13620H (10 Cores, 16 Threads, up to 4.90 GHz) | Parallel data preparation, BM25 indexing, FAISS IVF-PQ (CPU), Monte Carlo simulations, statistical tests |
| **Dedicated Graphics (dGPU)** | NVIDIA® GeForce RTX™ 4060 Laptop GPU (8,188 MiB GDDR6 VRAM, Ada Lovelace) | Single-worker GPU training, mixed-precision inference (bf16), Deepfake detection, Audio models |
| **Graphics Driver** | NVIDIA Windows Driver 592.82 (WSL DirectX/CUDA pass-through) | Pass-through CUDA 12.x/13.x compute directly to WSL2 |
| **System Memory (RAM)** | 16 GB DDR5-5600 MT/s (Dual Channel) | 12 GB allocated to WSL2; streaming JSON loaders (`ijson`) and mmap Arrow/parquet datasets |
| **Storage (Primary NVMe)** | 1 TB Western Digital PC SN560 PCIe NVMe SSD | D: Partition (188 GB free): Project repo, data manifests, cached models, execution workspace |
| **Backup Storage** | External / Secondary partition E: (88 GB free) | Target directory `E:\ares_backup\` for nightly mirrored rsync backups |

---

## 2. Windows-Side Configuration

### 2.1 WSL2 Resource Management (`%UserProfile%\.wslconfig`)
To prevent Windows memory starvation while giving WSL2 maximum capacity without risking OOM crashes:
```ini
[wsl2]
memory=12GB
processors=14
swap=16GB
swapFile=D:\\wsl\\swap.vhdx
```
* **Memory:** Capped at 12 GB (leaving 4 GB dedicated to Windows and host OS processes).
* **Processors:** 14 logical cores exposed to WSL2.
* **Swap:** 16 GB swap file explicitly allocated on partition `D:\wsl\swap.vhdx` to avoid exhausting `C:` drive storage.

### 2.2 Power & Thermal Profiles
* **OMEN Gaming Hub:** Set to **Performance Mode** (fans dynamically adjust for sustained boost clocks).
* **Windows Power Plan:** "Sleep: Never" and "Hibernate: Never" while plugged into AC power.
* **AC Adapter:** High-wattage barrel AC adapter connected for all computational workloads. Battery-only training is strictly forbidden.

---

## 3. WSL2 Environment Configuration

### 3.1 Base Toolchain Installation
Installed via `apt`:
```bash
apt-get update && apt-get install -y \
    build-essential git git-lfs tmux task-spooler ffmpeg rsync curl wget python3-pip python3-dev
```

### 3.2 Environment Variables (`~/.bashrc`)
```bash
export ARES_ROOT="/mnt/d/ares"
export ARES_DATA_ROOT="$ARES_ROOT/data"
export HF_HOME="$ARES_ROOT/cache/hf"
export TORCH_HOME="$ARES_ROOT/cache/torch"
export PIP_CACHE_DIR="$ARES_ROOT/cache/pip"
export UV_CACHE_DIR="$ARES_ROOT/cache/uv"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

### 3.3 Git Credential Integration
WSL2 git is configured to route through the host Windows Git Credential Manager:
```bash
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager.exe"
```

---

## 4. Verification Checklist

* [x] `nvidia-smi` detects GeForce RTX 4060 Laptop GPU with 8,188 MiB VRAM.
* [x] Host `.wslconfig` configured and active.
* [x] Backup directory `E:\ares_backup` created.
* [x] Git repository synchronized with remote `https://github.com/gurarpitzz/Ares-Cognitive-Computation`.
