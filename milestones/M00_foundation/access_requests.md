# ARES Rebuild — Restricted Dataset Access Requests & API Keys

**Milestone:** M00 (Step M00.8)  
**Date Generated:** 2026-10-13  
**Status:** READY FOR SUBMISSION BY HUMAN RESEARCHER  

This document logs all external dataset request forms and API keys required for Tier 1 and Tier 2 tracks in the ARES Master Implementation Plan (`plan/03_IMPLEMENTATION_PLAN.md` §4). Public datasets that require no agreement or credentials (LIAR, FEVER, AVeriTeC, Constraint, CoAID, PHEME, UPFD, ASVspoof 2019, Cresci, PAWS) are omitted here and directly acquired in M02.

---

## 1. Tier 1 Restricted Datasets (Core Benchmark)

| Dataset | Track | Official URL / Form | Information to Submit | Licence Summary | Expected Turnaround | Current Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FaceForensics++ (c23)** | T4 Video Deepfake | [TUM Form](https://github.com/ondyari/FaceForensics) | Academic affiliation, supervisor/PI name, project description ("ARES Misinformation Verification Benchmark") | Non-commercial academic research only. No redistribution of raw faces/videos. | 1–3 business days | `PENDING_HUMAN_SUBMISSION` |
| **Celeb-DF v2** | T4 Video Deepfake | [Celeb-DF Form](https://github.com/yuezunli/celeb-deepfakeforensics) | University email, academic request form | Non-commercial research only. | 2–5 business days | `PENDING_HUMAN_SUBMISSION` |
| **VoxCeleb1 (test trials)** | T5 Speaker Sim | [VoxCeleb Form](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | Academic registration via VGG portal | Research only. Test set lists only needed. | Instant / 24 hours | `PENDING_HUMAN_SUBMISSION` |
| **TwiBot-20** | T8 Bot Detection | [TwiBot GitHub / Form](https://github.com/BunsenFeng/TwiBot-20) | Fill out Google form linked in repository | Academic non-commercial. | 1–3 business days | `PENDING_HUMAN_SUBMISSION` |
| **MultiClaim** | T9 Claim Matching | [Zenodo MultiClaim](https://zenodo.org/records/7576508) | Access request via Zenodo with academic reason | Research only. | 2–4 business days | `PENDING_HUMAN_SUBMISSION` |

---

## 2. Tier 2 Extended Datasets (Generalisation & Robustness)

| Dataset | Track | Official URL | Information to Submit | Licence / Usage | Expected Turnaround | Current Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FakeAVCeleb** | T6 Audio-Visual | [FakeAVCeleb Form](https://github.com/DASH-Lab/FakeAVCeleb) | Academic form | Research only | 2–5 days | `PENDING_HUMAN_SUBMISSION` |
| **NewsCLIPpings / VisualNews** | T3 Image-Text | [VisualNews Request](https://github.com/g-luo/news_clippings) | Academic form | Non-commercial | 2–4 days | `PENDING_HUMAN_SUBMISSION` |
| **DFDC Preview** | T4 Video Deepfake | [DFDC Kaggle / Meta](https://www.kaggle.com/c/deepfake-detection-challenge) | Kaggle terms agreement | Competition terms | Instant upon agreement | `PENDING_HUMAN_SUBMISSION` |
| **WildDeepfake** | T4 Video Deepfake | [WildDeepfake Form](https://github.com/deepfakeinthewild/wild-deepfake) | Academic form | Research only | 2–4 days | `PENDING_HUMAN_SUBMISSION` |
| **DeeperForensics-1.0** | T4 Video Deepfake | [DeeperForensics Form](https://liming-jiang.com/projects/DrF1/DrF1.html) | Request agreement form | Academic only | 3–5 days | `PENDING_HUMAN_SUBMISSION` |
| **Snopes Corpus (UKP)** | T2 Evidence | [UKP Form](https://www.informatik.tu-darmstadt.de/ukp/research_67/data/) | Academic request | Research only | 1–3 days | `PENDING_HUMAN_SUBMISSION` |
| **IFND / FactDrill** | T1/T10 Indian News | [IFND / Authors Request](https://github.com/rksahu1/IFND) | Academic email request | Research only | 3–7 days | `PENDING_HUMAN_SUBMISSION` |
| **MOCHEG / Factify 2** | T3 Image-Text | [De-Factify Workshop](https://defactify.com/) | Workshop form / Zenodo | Non-commercial | 1–3 days | `PENDING_HUMAN_SUBMISSION` |
| **VERITE** | T3 Image-Text | [VERITE GitHub](https://github.com/IDT-ITI/VERITE) | Request form | Research only | 1–3 days | `PENDING_HUMAN_SUBMISSION` |
| **TwiBot-22** | T8 Bot Detection | [TwiBot-22 Form](https://github.com/BunsenFeng/TwiBot-22) | Academic request | Research only (subgraph) | 2–4 days | `PENDING_HUMAN_SUBMISSION` |

---

## 3. External API Keys and Model Licences

| Service / Model | Role | Portal / Sign-up URL | Key / Action Required | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Google Fact Check Tools API** | T11 Temporal External Validation | [Google Cloud Console](https://developers.google.com/fact-check/tools/api) | Create free API Key for Fact Check Tools API | `PENDING_HUMAN_KEY` |
| **Hugging Face Hub** | Models (Qwen2.5, Mistral, DeBERTa) | [HuggingFace.co](https://huggingface.co/) | Accept gate licenses for Qwen2.5-7B-Instruct & Mistral-7B-Instruct-v0.3; export `HF_TOKEN` | `PENDING_HUMAN_TOKEN` |
| **ReliefWeb API** | M16 Assam Floods Ground Truth | [ReliefWeb API](https://apidoc.rwlabs.org/) | Register application name (free token) | `PENDING_HUMAN_TOKEN` |
| **Web Search API** | Showcase / T11 Live Trace | Tavily / Serper / Bing Search | Free tier API key for live search corroboration | `OPTIONAL` |

---

## 4. Human Action Instructions
1. Submit the 5 **Tier 1** forms immediately (TUM FaceForensics++, Celeb-DF, VoxCeleb, TwiBot-20, MultiClaim).
2. Generate the **Google Fact Check Tools API Key** and accept the Hugging Face model licenses.
3. Save environment tokens into `d:\ares\.env` (which is gitignored) as:
   ```bash
   HF_TOKEN="your_hf_token_here"
   GOOGLE_FACTCHECK_API_KEY="your_api_key_here"
   RELIEFWEB_APPNAME="ares_rebuild"
   ```
