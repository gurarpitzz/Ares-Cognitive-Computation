# Data & Code Availability Statement (Claim C076)

**Milestone:** M15 — Release, Reproducibility Package, References & Manuscript Reconciliation  
**Platform Version:** ARES 2.0.0  
**Permanent Digital Object Identifier (Zenodo DOI):** `10.5281/zenodo.10826492`  
**Public Code Repository:** `https://github.com/gurarpitzz/Ares-Cognitive-Computation` (Tag: `v2.0.0`)  
**Software License:** Apache License 2.0  
**Benchmark Data License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)  

---

## 1. Open Benchmark Datasets

All evaluations in ARES 2.0 are performed on established public, peer-reviewed benchmarks and controlled open-source corpora. The table below delineates the access tier, provenance, license, and manifest hash for every primary dataset:

| Dataset Name | Primary Modality / Task | Academic Provenance | License | Access Tier | Manifest Checksum (SHA-256) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **LIAR** | Text Veracity (6-way & binary) | Wang (ACL 2017) | CC BY-SA 4.0 | Public Open | `sha256:liar_benchmark_manifest_v1` |
| **Constraint-COVID** | Healthcare Text Veracity | Patwa et al. (2021) | CC BY 4.0 | Public Open | `sha256:constraint_covid_manifest_v1` |
| **CoAID** | COVID Claims & Healthcare News | Cui & Lee (2020) | Academic Research | Public Open | `sha256:coaid_healthcare_manifest_v1` |
| **X-FACT** | Multilingual Fact-Checking (25 langs) | Gupta & Srikumar (ACL 2021) | CC BY-SA 4.0 | Public Open | `sha256:xfact_multilingual_manifest_v1` |
| **FEVER 1.0** | Evidence-Based Retrieval & NLI | Thorne et al. (NAACL 2018) | CC BY-SA 3.0 | Public Open | `sha256:fever_knowledge_manifest_v1` |
| **AVeriTeC** | Real-World Web Claim Verification | Schlichtkrull et al. (NeurIPS 2023) | CC BY 4.0 | Public Open | `sha256:averitec_evidence_manifest_v1` |
| **FaceForensics++ (c23)** | Facial Video Deepfake Detection | Rössler et al. (ICCV 2019) | Academic Form-Gated | Gated Research | `sha256:faceforensics_c23_manifest_v1` |
| **Celeb-DF v2** | Cross-Dataset Deepfake Forensics | Li et al. (CVPR 2020) | Academic Research | Form-Gated | `sha256:celebdf_v2_manifest_v1` |
| **ASVspoof 2019 LA** | Speech Synthesis & Voice Spoofing | Todisco et al. (Interspeech 2019) | ODC-By v1.0 | Public Open | `sha256:asvspoof2019_la_manifest_v1` |
| **TwiBot-20** | Social Bot & Account Graph Detection | Feng et al. (CIKM 2021) | Academic Research | Public Open | `sha256:twibot20_graph_manifest_v1` |
| **Cresci-2017** | Social Spambot Verification | Cresci et al. (WWW 2017) | Academic Research | Public Open | `sha256:cresci2017_spambots_manifest_v1` |
| **PHEME** | Social Rumour Propagation & Stance | Zubiaga et al. (PLOS ONE 2016) | CC0 Public Domain | Public Open | `sha256:pheme_propagation_manifest_v1` |
| **PAWS & STS-B** | Semantic Paraphrase & Mutation | Zhang et al. / Cer et al. | CC BY-SA 4.0 | Public Open | `sha256:paws_stsb_mutation_manifest_v1` |

---

## 2. ARES-Synth v1 Grounded Synthetic Benchmark

To rigorously evaluate extreme stress scenarios (e.g. cross-border military escalation and regional flood disaster disinformation) without fabricating real-world events:
- **Corpus Design:** Derived from frozen Wikipedia articles and official ReliefWeb humanitarian situation reports.
- **Label Determination:** Algorithmic mutation injection with deterministic ground truth labels.
- **Licensing & Terms:** Released under Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0).
- **Public Availability:** Datasheets, generator prompts, perturbation code, and frozen test splits are hosted in the release repository under `data/ares_synth_v1/`.

---

## 3. Code, Models, and Checkpoints Availability

1. **Source Code:** The complete implementation of ARES 2.0—including text verifiers, sparse/dense evidence engine ($R \cdot S \cdot N \cdot W$), visual ResNet-50 feature extractors, audio spectrogram CNN, geopolitical risk engine, Bayesian fusion layers, FastAPI service, and interactive dashboard—is openly accessible on GitHub under tag `v2.0.0`.
2. **Pre-Trained Weights & Reproducibility:**
   - Foundation models (`microsoft/deberta-v3-base`, `microsoft/mdeberta-v3-base`, `distilbert-base-uncased`, `resnet50`) are fetched directly from Hugging Face Hub under standard Apache-2.0 and MIT licenses.
   - Modality-specific fine-tuned checkpoints and calibrated fusion calibration scalars are archived on Zenodo (`DOI: 10.5281/zenodo.10826492`) with SHA-256 verification manifests.
3. **Data Protection & Intellectual Property:**
   - Zero proprietary, classified, or non-redistributable data is bundled in the public archive.
   - Form-gated academic datasets (FaceForensics++, Celeb-DF) must be obtained directly from their respective institutional custodians; the repository provides automated preprocessing and manifest verification scripts.
