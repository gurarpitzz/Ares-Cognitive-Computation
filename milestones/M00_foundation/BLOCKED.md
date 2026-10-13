# BLOCKED — M00 Step M00.8: Restricted Dataset Access Requests & API Keys

### What I need from the human:
1. Submit external access request forms for the 5 Tier-1 restricted datasets:
   - **FaceForensics++** (TUM academic form)
   - **Celeb-DF v2** (academic request)
   - **VoxCeleb1** (VGG user registration)
   - **TwiBot-20** (Google Form)
   - **MultiClaim** (Zenodo request)
2. Generate a free **Google Fact Check Tools API Key** and accept Hugging Face gate licenses for `Qwen/Qwen2.5-7B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`.
3. Save tokens into `d:\ares\.env`.

### Why:
These datasets are restricted by academic licensing and cannot be scraped or downloaded autonomously without human registration.

### Options:
- **Option 1 (Recommended):** Submit forms today while M00 and M01 (theory proofs and Monte Carlo simulations) proceed on the workstation. By the time M02 (data curation) begins, approvals will be coming in.
- **Option 2:** If access is denied or delayed for any specific dataset, evaluate on alternative open benchmarks (e.g. In-the-Wild for audio, UPFD for social propagation) and record the substitution in `M02/README.md` as pre-registered in the plan.

### What I'll do meanwhile:
Proceed seamlessly with M00 environment lockfile, integrity tests, VRAM fit benchmarking, kill-and-resume tests, and M01 mathematical proofs. No scientific execution in M00 or M01 requires these restricted datasets.
