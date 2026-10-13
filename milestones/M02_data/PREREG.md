# M02 Pre-Registration — Data Acquisition, Curation, Splits & Leakage Audit

**Milestone:** M02 — Data Acquisition, Curation, Splits & Leakage Audit  
**Date:** 2026-10-13  
**Status:** PRE-REGISTERED  
**Target Ledger Claims:** C032 (dataset volumes), C088 (VoxCeleb trial evaluation), C089 (X-FACT multilingual coverage)  
**Primary Verification Metrics:** Manifest SHA-256 integrity, split leakage rate (exact & near duplicate overlap), class balance, storage peak volume  
**Compute & Storage:** RTX 4060 Laptop (Intel Core i7-13620H, 16 threads, 202 GB free disk on D:)  

---

## 1. Scope & Objective
Acquire, canonicalize, validate, and hash every public Tier-1 dataset in the ARES benchmark registry, followed by Tier-2 datasets processed with strict disk quota discipline. Pre-register all label mapping collapses, grouped split policies, and leakage audit rules **before** any evaluation or downstream training occurs in M03–M12.

---

## 2. Canonical Data Schema
Every dataset must be converted into Parquet format in `data/processed/<name>/` conforming to the schema:
- `item_id` (string, primary key, deterministic content/dataset hash)
- `split` (categorical: `train`, `val`, `test`, `unlabeled`)
- `label_raw` (string, original source label verbatim)
- `label` (string or integer, pre-registered canonical label)
- `text` (string, nullable, UTF-8 normalized text of the claim / article)
- `lang` (string, ISO 639-1 language code, e.g. `en`, `es`, `de`)
- `date` (string, nullable, ISO 8601 date timestamp)
- `group_id` (string, nullable, speaker / publisher / identity / event ID for grouped splits)
- `media_path` (string, nullable, relative path to media asset)
- `meta` (string, serialized JSON dictionary of non-leakage auxiliary metadata)

---

## 3. Pre-Registered Label Mappings

### 3.1 LIAR (PolitiFact 6-way & Binary)
- **Primary Binary Collapse:**
  - `TRUE` (1): `{"true", "mostly-true", "half-true"}`
  - `FALSE` (0): `{"barely-true", "false", "pants-fire"}`
- **Sensitivity Variant:** Exclude `half-true` to test boundary ambiguity:
  - `TRUE` (1): `{"true", "mostly-true"}`
  - `FALSE` (0): `{"barely-true", "false", "pants-fire"}`
- **Native 6-way:** `{"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5}`.

### 3.2 FEVER (Fact Extraction and VERification)
- Native 3-way: `SUPPORTS` (0), `REFUTES` (1), `NOT ENOUGH INFO` (2).

### 3.3 Constraint-COVID19 (English)
- Native 2-way: `real` (1), `fake` (0).

### 3.4 CoAID (COVID-19 Healthcare Misinformation)
- Native 2-way: `real` (1), `fake` (0), grouped by source web domain and article title.

### 3.5 X-FACT (Multilingual Fact-Checking)
- Native 7-way: `true`, `mostly true`, `partly true`, `mostly false`, `false`, `unverifiable`, `other`.
- Binary collapse: `true`/`mostly true` $\to$ TRUE (1); `false`/`mostly false` $\to$ FALSE (0); others excluded from binary benchmark.

### 3.6 PHEME (Rumour and Verification)
- Native 3-way: `true` (1), `false` (0), `unverified` (2).

### 3.7 UPFD (User Preference-aware Fake News Detection)
- Native 2-way: `real` (1), `fake` (0), Politifact and Gossipcop graphs.

### 3.8 ASVspoof 2019 / 2021 LA (Logical Access Audio)
- Native 2-way: `bonafide` (1), `spoof` (0).

---

## 4. Split Policy and Manifests
1. **Official Splits Mandate:** If a public canonical benchmark provides official train/val/test splits (LIAR, FEVER, Constraint, X-FACT, ASVspoof, PAWS, STS-B), those splits **must** be preserved bit-for-bit to enable apples-to-apples comparison with published literature.
2. **In-House Splits:** For datasets lacking official splits (CoAID, Cresci-2017), construct grouped stratified splits:
   - Ratios: 70% train / 10% val / 20% test.
   - Pinned seed: `20261013`.
   - Grouping: by speaker / publisher / news domain so that no group appears in both train and test.
3. **PHEME:** Leave-one-event-out folds (5 events: Charlie Hebdo, Ferguson, Ottawa shooting, Sydney siege, Germanwings crash).
4. **Manifest Hashing:** Every split manifest in `ares_v2/data_manifests/<name>.csv` lists `item_id`, `split`, `label`, `content_sha256`. The overall manifest SHA-256 is recorded in `manifests.json`.

---

## 5. Leakage Audit Pre-Registration
1. **Exact Deduplication:** Normalised text SHA-256 hashes must not overlap between train and test splits.
2. **Near-Duplicate Detection:**
   - MinHash LSH (character 5-grams, Jaccard similarity $\ge 0.8$).
   - Embedding cosine similarity $\ge 0.95$ using `sentence-transformers/all-MiniLM-L6-v2`.
3. **Cross-Dataset Overlap:** Measure pairwise PolitiFact claim overlap between LIAR, X-FACT, UPFD, and AVeriTeC.
4. **Metadata Leakage Prevention:** Label-revealing fields (e.g. LIAR-PLUS justification, PolitiFact ruling text, speaker party bias) are excluded from the model input text.

---

## 6. Storage Budget and Quota Discipline
- Peak disk usage on `D:` must remain $\le 150$ GB at all times.
- Before downloading any large dataset, execute storage check and log to `milestones/M02_data/tables/storage_projection.csv`.
- Process heavy media datasets one at a time: download $\to$ extract crops/features $\to$ verify hashes $\to$ delete raw source archives if retention policy is `delete-after-derive`.
