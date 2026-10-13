# M02 — Data Acquisition, Curation, Splits & Leakage Audit

**Objective:** Acquire every dataset in the registry (`03_IMPLEMENTATION_PLAN.md §4`). Tier 1 is required. Tier 2 means "all relevant public data", processed one at a time. Convert each dataset to a canonical schema with fixed, hashed split manifests, pre-register the label mappings, and run a leakage audit **before** any model sees the data. Also snapshot the source documents for M16.
**Ledger IDs:** C032 (dataset volumes), C088 (VoxCeleb misuse), C089 (DeFacto-X / X-FACT languages).
**Depends on:** M00.
**Compute:** RTX 4060 Laptop.
- CPU/disk-bound for text and social data.
- About 12 GPU-h for face extraction (FF++/Celeb-DF and the Tier-2 video subsets).
- Peak disk must stay ≤ 150 GB on D: (`scripts/disk_report.sh` before every download).
**Output dir:** `milestones/M02_data/`

---

## Steps

### M02.1 — Data registry
- **Action:** `configs/data/registry.yaml`, one entry per dataset with fields `name, tier (1|2), track, version, homepage, download_url(s), licence, redistribution (yes|ids-only|no), access (open|form|request), citation (BibTeX key), expected_counts (per split, per label, from the dataset paper), est_size_gb, measured_size_gb, local_path, raw_retention (keep|delete-after-derive)`.
- **Done when:**
  - Every Tier-1 and Tier-2 dataset in the plan's registry is listed.
  - Expected counts are copied from the original papers, with the citation.
  - A storage projection (`tables/storage_projection.csv`) shows that peak use stays ≤ 150 GB under the planned processing order.

### M02.2 — Downloaders (open datasets)
- **Action:** `scripts/data/download_<name>.py` for each open dataset: resumable download, SHA256 of every file, retrieval timestamp written to `data/raw/<name>/RETRIEVAL.json`. **Fail loudly** on any error (no fallbacks). Large JSON is parsed by streaming (`ijson`) into parquet shards; never `json.load` a multi-GB file in 12 GB of WSL RAM.
- **Order (Tier 1):** LIAR, FEVER (+ wiki-pages), AVeriTeC (+ dev/test knowledge stores only), Constraint-COVID (English), CoAID, X-FACT, PHEME, UPFD (PyG), Fakeddit (text + a fixed image subset, see M02.8), ASVspoof 2019 LA, ASVspoof 2021 LA eval, In-the-Wild, Cresci-2017, PAWS, STS-B, CheckThat! 2021/22 Task 2, ClaimBuster dataset, QuanTemp, LLMFake, Lin et al. 2023 domain ratings, GeoNames (allCountries or cities500), CoNLL-2003 (licence permitting) or a WikiNER alternative.
- **Order (Tier 2, open):** MultiFC, CLIMATE-FEVER, SciFact, PUBHEALTH, HealthVer, COVID-Fact, Twitter15/16, RumourEval 2019, WELFake, ISOT, Constraint Hindi, ASVspoof 2021 DF eval (seeded subset), WaveFake (seeded subset), Fake-or-Real, Cresci-2015, the Botometer repository sets.
- **Done when:** each dataset's file hashes are recorded and its counts printed; `disk_report.sh` output is logged after each download.

### M02.3 — Restricted datasets (after access is granted)
- **Action:** Same as M02.2.
  - **Tier 1:** FF++ (c23 + official split JSONs), Celeb-DF v2 (test list), VoxCeleb1 (test trials + test audio only), TwiBot-20, MultiClaim.
  - **Tier 2:** FakeAVCeleb, NewsCLIPpings, MOCHEG, Factify 2, VERITE, Snopes (UKP), IFND, FactDrill, DFDC preview, WildDeepfake, DeeperForensics-1.0 (documented subset).
  - Until access arrives, keep a `BLOCKED.md` entry per dataset and continue.

### M02.3b — Tier-2 processing discipline
- **Action:**
  - Process Tier-2 datasets **one at a time**: download → canonicalise → manifest → (for media) derive features or crops → verify hashes → delete the raw data if `raw_retention=delete-after-derive`.
  - Subsets (WaveFake, ASVspoof 2021 DF, DeeperForensics, DFDC preview, NewsCLIPpings) are **seeded and stratified** by attack/vocoder/manipulation. The subset rule is pre-registered in PREREG, and subset IDs are released.
  - A Tier-2 dataset that is skipped (access denied, storage, licence) is listed in the README with its reason.
- **Done when:** `tables/tier2_status.csv` gives, for every Tier-2 dataset: status (acquired / subset / skipped + reason), size, and subset rule.

### M02.4 — Canonical schema & loaders
- **Action:** `src/ares/data/<name>.py` → Parquet in `data/processed/<name>/`: required columns `item_id, split, label_raw, label, text (nullable), lang, date (nullable), group_id (speaker / event / identity / source; used for grouped splits), media_path (nullable), meta (JSON)`. Validate with a pydantic/pandera schema.
- **Done when:** `python -m ares.data.<name> --validate` passes for all acquired datasets.

### M02.5 — Label mappings (pre-registered)
- **Action:** `configs/data/label_maps.yaml` + `milestones/M02_data/PREREG.md`. Required decisions:
  - **LIAR binary (primary):** {true, mostly-true, half-true} → TRUE; {barely-true, false, pants-fire} → FALSE. **Sensitivity:** exclude half-true. Also keep 6-way.
  - FEVER: SUPPORTS / REFUTES / NOT ENOUGH INFO (native 3-way).
  - AVeriTeC: Supported / Refuted / Not Enough Evidence / Conflicting (native 4-way; plus a binary subset S vs R).
  - X-FACT: native 7 labels + binary collapse (document).
  - PHEME: rumour veracity (true/false/unverified) for rumours; document.
  - Fakeddit: 2-way and 6-way native.
  - ASVspoof: bonafide vs spoof. FF++: real vs fake (per manipulation tracked).
  - **Tier-2 maps:**
    - MultiFC: per-site label → {TRUE, FALSE, drop}; the table is committed before use.
    - PUBHEALTH: true/false/mixture/unproven → binary on true/false, plus a 4-way run.
    - CLIMATE-FEVER and SciFact: native S/R/NEI.
    - Twitter15/16: native 4-way.
    - IFND and FactDrill: native, with a documented binary collapse.
    - WELFake/ISOT: binary. ISOT must also report the shortcut-stripped variant (remove the "(Reuters)" dateline and the source boilerplate).
- **Done when:** PREREG committed before any training.

### M02.6 — Split policy & manifests
- **Action:** Use official splits wherever they exist (LIAR, FEVER, AVeriTeC, X-FACT, Constraint, ASVspoof, FF++, PAWS, STS-B, CheckThat). For datasets without official splits (CoAID, Cresci, Fakeddit subset): grouped stratified split 70/10/20 by `group_id`, seed `20261013`. PHEME: **leave-one-event-out** folds. Write `ares_v2/data_manifests/<name>.csv` (item_id, split, label, content sha256) + manifest hash.
- **Done when:** manifests committed; split sizes in `tables/split_sizes.csv` match official numbers (LIAR: 10,269 / 1,284 / 1,283).

### M02.7 — Leakage audit
- **Action:** `scripts/data/leakage_audit.py` → `results/leakage_report.json` + `tables/leakage_summary.csv`:
  1. Exact duplicates within and across splits (normalised text hash).
  2. Near-duplicates: MinHash LSH (char 5-grams, Jaccard ≥ 0.8) **and** embedding cosine ≥ 0.95 (all-MiniLM-L6-v2) across train/test.
  3. **Cross-dataset contamination:** PolitiFact-sourced claims appear in LIAR, X-FACT, FakeNewsNet/UPFD, AVeriTeC, MultiClaim. Measure overlap pairwise. Needed for cross-domain tests (M12) and claim memory (M06).
  4. Group overlap: FF++ identities, ASVspoof speakers, PHEME events, Cresci/TwiBot users.
  5. Label-revealing fields: LIAR speaker/party/context columns, LIAR-PLUS justifications, Fakeddit metadata. Document which fields are allowed per experiment (default: **text only** for text tracks).
- **Decision rule (pre-registered):** remove cross-split near-duplicates from the *test* side for in-house splits; for official splits, keep the official test set and report results both with and without near-duplicates.
- **Done when:** report produced; decisions recorded in PREREG/README.

### M02.8 — Media processing
- **Action:**
  - **FF++ / Celeb-DF:**
    - Per video, sample 32 frames uniformly.
    - Detect faces on the GPU (RetinaFace/insightface or MTCNN), in batches that fit 8 GB.
    - Crop with a 1.3× margin to 299×299 (store 380×380 too if EfficientNet-B4 is used). Save JPEG quality 95 (or PNG if disk allows), with a per-frame hash.
    - Log the face-detection failure rate **per class and per manipulation**.
    - **Delete raw videos once the crops are verified and hashed** (required by the storage budget), and record the deletion in the manifest.
    - Celeb-DF: process the official test list only.
  - **Fakeddit:**
    - Download a fixed, stratified, seeded subset of image posts from the official splits: **60k train / 6k val / 6k test** (6-way labels stratified).
    - Record dead links, hash the images, store the IDs.
    - Write the rationale (storage budget) into PREREG.
  - **ASVspoof:** record utterance durations and leading/trailing-silence durations (needed for the known silence shortcut check in M08).
- **Done when:** processed media + hashes + failure-rate tables exist.

### M02.9 — Social/propagation data
- **Action:**
  - PHEME: parse source tweets, reply trees, timestamps and user metadata into tables.
  - UPFD: load via `torch_geometric.datasets.UPFD` (politifact, gossipcop; feature types bert/profile/spacy/content).
  - Cresci-2017/TwiBot-20: user feature tables + tweet timelines + follow edges, streamed with `ijson` into parquet.
  - Tier 2: Twitter15/16 and RumourEval trees (report the coverage of still-available text); Cresci-2015 and the Botometer sets.
- **Done when:** tables validated; counts in dataset cards.

### M02.10 — Snapshot infrastructure for external sources
- **Action:** `src/ares/data/snapshot.py`: `SnapshotClient(name)` wrapping HTTP GET/POST with request canonicalisation → hash → stored response (`data/snapshots/<name>/<hash>.json` with timestamp, URL, params, status). Modes: `record` (live + store) and `replay` (store only; error on miss). Experiments **must** run in replay mode. Used for web search, Wikidata SPARQL, GDELT, Fact Check API, Factbook.
- **Done when:** unit tests: record → replay returns identical bytes; replay miss raises.

### M02.10b — M16 source-document snapshot (input for ARES-Synth)
- **Action:** Use `SnapshotClient` in record mode to snapshot the source documents for the M16 scenarios. Record a fixed **revision ID / retrieval date** for each:
  - **S1 India–China border:** Wikipedia articles on the 2020–2021 China–India skirmishes, the 2020 Galwan Valley clash, and the 2022 Yangtse (Tawang) clash. Pin them via `oldid` at a fixed revision date, chosen in PREREG.
  - **S1, also:** official statements from India's MEA/PIB and China's MFA, stored as URL + hash + extracted text.
  - **S2 Assam floods 2022:** ASDMA daily flood reports (PDF → text), ReliefWeb situation reports (ReliefWeb API), and the Wikipedia article on the 2022 Assam floods.
  - **S3 (optional) Indian general election 2024:** ECI results pages and Wikipedia.
  - **Distractors:** same-topic documents about different events and years (for example 2017 Doklam, Assam floods 2012/2020), for hard negatives and retrieval noise.
- **Files:** `data/synthetic_sources/<scenario>/` plus `ares_v2/data_manifests/synth_sources.csv` (url, revision/date, sha256, licence, redistributable y/n).
- **Done when:** every source has a hash and licence; the manifest is committed; nothing in this step uses an LLM.

### M02.11 — Dataset cards
- **Action:** `milestones/M02_data/dataset_cards/<name>.md`: source, licence, version, counts per split × label, languages, date range, known issues, our label mapping, allowed fields, leakage findings.
- **Ledger:** C032 → actual counts (status REVISED, with a table of the real volumes). C088 → note VoxCeleb used only for speaker verification; ASVspoof for spoofing. C089 → X-FACT language count measured; DeFacto-X marked REMOVE unless identified.

### M02.12 — Literature reference ranges (for audit plausibility only)
- **Action:** `milestones/M02_data/sota_reference.csv`: dataset, split, metric, value, model, citation (full, verified by web search), notes. At least: LIAR (6-way, binary), FEVER (label acc / FEVER score), AVeriTeC (baseline + shared-task best), Constraint-COVID, X-FACT, PHEME (LOEO), UPFD, Fakeddit, FF++ c23, Celeb-DF cross, ASVspoof 2019 LA (LCNN, AASIST), VoxCeleb1-O (ECAPA), TwiBot-20, Cresci-2017, CheckThat Task 2, PAWS.
- **Rule:** these are **not targets**; they exist so the auditor can flag implausible results.

---

## Audit gate — milestone-specific red flags
- RF1: Split sizes and label distributions match the original papers (cite them). Any mismatch explained.
- RF2: No silent drops. Every removed item is logged with a reason.
- RF3: FF++ splits are identity-disjoint per official JSON; frames from one video never span splits.
- RF4: PHEME evaluated with leave-one-event-out folds, not random splits.
- RF5: Cross-dataset PolitiFact overlap quantified.
- RF6: Face-detection failure rates not systematically higher for one class (if so, flagged as bias risk).
- RF7: Snapshot client in replay mode makes no network calls (test with network disabled).
- RF8: Licences respected; nothing restricted committed to git.
- RF9: Every Tier-2 dataset is accounted for in `tier2_status.csv`; none was skipped silently. Subset rules are pre-registered and seeded.
- RF10: Raw-data deletions happened only after the derived artefacts were hash-verified. Every deletion is recorded. The peak disk log stays ≤ 150 GB.
- RF11: ISOT/WELFake shortcut tokens are documented, and the stripped variant exists.

## Exit criteria
- [ ] All open Tier-1 datasets acquired, hashed, processed, validated; restricted ones acquired or BLOCKED with status.
- [ ] Tier-2 status table complete (acquired / subset / skipped + reason).
- [ ] M16 source snapshot manifest committed.
- [ ] Label maps + split policy pre-registered and committed before use.
- [ ] Manifests committed; leakage report complete with decisions.
- [ ] Dataset cards + SOTA reference table complete.
- [ ] Ledger C032, C088, C089 updated.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m02-complete`.
