# M16 — ARES-Synth: Grounded Synthetic Stress Benchmarks (India–China border, Assam floods, Indian election 2024)

**Objective:** Build controlled test sets with **known ground truth** for the manuscript's scenario claims, for which no public labelled data exists:
- India–China border (C021);
- Assam floods 2022 (C022);
- Indian election (C020-India).

Claims are generated from **frozen, real source documents**. TRUE and FALSE labels are set by **deterministic code** (recorded perturbations), not by an LLM. An LLM only rewrites the surface wording. Every item is validated automatically, checked for shortcuts, and audited by a human before the set is **frozen**. Also produce mutation chains with known change points for M06.

The benchmark is a **controlled complement** to real data. It is never presented as real-world performance.

**Ledger IDs:** C020 (India part), C021, C022; supplies M06 (C011, C012, C086) and M12 (C023, C059).
**Depends on:**
- M02: source snapshots (M02.10b), `SnapshotClient`, manifests.
- M03: claim-only TF-IDF/DeBERTa training code, used here only as diagnostics.
- M04: the NLI interface. The validator *model* is deliberately different; see M16.6.

**Execution order:** after M04, before M05/M06. It **must be frozen (tag `ares-synth-v1`) before M11 starts.**
**Compute:** RTX 4060 Laptop, ≈ 6–8 GPU-h:
- Mistral-7B-Instruct-v0.3 in 4-bit for surface realisation;
- the NLI validators;
- the claim-only diagnostic classifiers.

Everything else runs on the CPU.
**Output dir:** `milestones/M16_synthetic_stress/`; data in `<ARES_ROOT>/data/synthetic/ares_synth_v1/`.

> **Integrity rules specific to M16** (they add to Protocol §1):
> 1. Everything is labelled SIMULATED / `data_kind="synthetic_grounded"`.
> 2. **Test-only:** no ARES component is trained, tuned or thresholded on it.
> 3. **No ARES module is run on ARES-Synth before the freeze.** This stops the benchmark from being shaped around the system.
> 4. **The generator family ≠ the evaluated LLM family.** Mistral generates; Qwen is the ARES LLM member.
> 5. **The validator NLI ≠ the ARES N-operator model.**
> 6. **Contested facts are never used as TRUE or FALSE.**
> 7. No private individuals, and no inflammatory or derogatory wording.

---

## Pre-registration (commit `PREREG.md` before generating anything)

### Scenarios
| ID | Scenario | Window | Primary sources (snapshotted in M02.10b) |
|---|---|---|---|
| S1 | India–China border | 2020-05-01 → 2023-01-31 (Galwan 2020, Pangong/eastern Ladakh disengagement, Yangtse/Tawang 2022) | Wikipedia (pinned `oldid`), MEA/PIB statements, China MFA statements |
| S2 | Assam floods 2022 | 2022-04-01 → 2022-10-31 | ASDMA daily flood reports, ReliefWeb situation reports, Wikipedia |
| S3 (optional) | Indian general election 2024 | 2024-03-16 → 2024-06-10 | ECI results and schedule, Wikipedia |

### Labels (evidence semantics, relative to the frozen evidence corpus)
| Label | Meaning |
|---|---|
| **TRUE** | Entailed by at least one evidence-corpus sentence. Construction: a faithful rewording of an extracted atomic fact. |
| **FALSE** | Contradicted by at least one evidence-corpus sentence. Construction: a deterministic perturbation of an atomic fact. |
| **NEI** | Not decidable from the evidence corpus. Two sources: (a) a TRUE fact whose source sentences were **withheld** from the evidence corpus (it is real, but there's no evidence for it); (b) a **fabricated event** that combines scenario entities with an event that occurs nowhere in the sources. (b) is reported separately as the *false-acceptance rate*: how often a system says TRUE. |
| **CONTESTED** | Facts on which the sources disagree. Examples: casualty figures, who initiated a clash, territorial and sovereignty statements, troop numbers. **Excluded from all T/F/NEI metrics.** Used only as an abstention test: the ideal output is "Uncertain". |

The contested-facts list for S1 is written **before** fact extraction, from the source disagreements. An extracted fact that matches it is routed to CONTESTED automatically.

### Perturbation types (FALSE unless noted)
`entity_swap`, `location_swap`, `number_change`, `date_shift`, `negation`, `misattribution`, `out_of_context` (a real fact moved to another event or year), `fabricated_event` (→ NEI-fabricated).

### Targets per scenario, after filtering
- ≥ 600 TRUE, ≥ 600 FALSE (≥ 80 per FALSE type), ≥ 300 NEI (withheld + fabricated), and all CONTESTED items found.
- 300 mutation chains of length 6–10, each with 0, 1 or 2 mutation points (balanced).
- If a target can't be met, report the shortfall. Never pad with unvalidated items.

### Acceptance thresholds
| Check | Threshold |
|---|---|
| Slot preservation | 100 % of kept items |
| Validator NLI, TRUE | mean entailment ≥ 0.90 from **both** validators against the source sentence |
| Validator NLI, FALSE | entailment < 0.10 from both validators, **and** the slot-difference check passes |
| Validator NLI, NEI | no evidence-corpus sentence (BM25 top-50) has entailment ≥ 0.5 or contradiction ≥ 0.5 from either validator |
| Claim-only shortcut ceiling | Binary T vs F macro-F1 ≤ 0.60 (5-fold CV) for TF-IDF+LR and for DeBERTa-v3-base claim-only |
| Human audit | Accuracy against construction labels ≥ 0.90, with the Wilson 95 % lower bound ≥ 0.85; unsafe/unnatural flags ≤ 5 % |
| Iterations | At most **2** rebuild iterations. Each one is a new pre-registered version (v1a, v1b), and every version's numbers are reported |

### Evaluation plan (executed in M12.6b, not here)
- Systems and metrics as listed in M12.6b.
- Bootstrap CIs clustered by source fact. Paraphrases of one fact are not independent, so cluster at the `fact_id` level.

---

## Steps

### M16.1 — Scenario cards
- **Action:** Write `milestones/M16_synthetic_stress/scenario_cards/S{1,2,3}.md`, each containing:
  - the window and the source list (with M02.10b manifest rows);
  - the **entity lists** (organisations, official roles, places), each with a Wikidata QID;
  - the **gazetteer**: GeoNames IDs and coordinates for every location (for example Galwan Valley, Pangong Tso, Yangtse/Tawang, the Assam districts);
  - the S1 contested-facts list, with the conflicting sources cited;
  - a **public-figure allow-list**: only named officials who appear in the sources. No private individuals.
- **Done when:** the cards are committed. Every location has a GeoNames ID, every entity a QID, and the contested list has citations.

### M16.2 — Freeze the evidence corpus
- **Action:**
  - Segment the M02.10b snapshots into sentences: `doc_id`, `sent_id`, text, source, date, licence.
  - Using a seeded draw (seed `20261013`), mark **15 % of fact-bearing sentences as WITHHELD**. These are removed from the evidence corpus and become the source of NEI-withheld items.
  - Add the distractor documents (other years/events).
  - Write `evidence_corpus.parquet` + `withheld.parquet` with their SHA256 values.
- **Done when:** the hashes are recorded; no withheld sentence (or near-duplicate, cosine ≥ 0.95) remains in the evidence corpus.

### M16.3 — Atomic fact extraction
- **Action:** `src/ares/synth/facts.py`. From each fact-bearing sentence, extract tuples `(fact_id, subject, relation, object, number+unit, date, location_geonames_id, attributed_source, sent_id)`.
  - **Independence from ARES:** use an extraction implementation **independent of the M04 logic layer**, for example spaCy NER + `dateparser` + regex numerics here if M04 uses quantulum3, or the reverse. Document the choice. This stops the benchmark from being built around the logic layer's own parser.
  - An LLM may *propose* candidate tuples. A tuple is accepted only if every slot value appears in the source sentence, verbatim or after normalisation. This check is done by code.
  - Route every tuple that matches the contested list to CONTESTED.
- **Human spot-check:** 100 random tuples, checked against their source sentences by the agent and listed for the human. Report the error rate.
- **Done when:** `facts.parquet` has its counts per scenario and relation type, and the spot-check error rate is reported.

### M16.4 — Deterministic perturbation engine (labels come from here)
- **Action:** `src/ares/synth/perturb.py`, one pure function per type, unit-tested:
  - **`entity_swap`:** replace the subject or object with a **different entity of the same type** from the scenario entity list. Never with an entity that also satisfies the fact in the corpus (checked against all tuples).
  - **`location_swap`:** replace with a gazetteer location in the same region, at a distance of ≥ 50 km (S1) or a different district (S2). Record the gold and swapped GeoNames IDs.
  - **`number_change`:** multiply by a factor drawn from {0.25, 0.5, 2, 3, 10} with seeded RNG, or add an offset. The relative difference must be ≥ 25 % and beyond any rounding in the sources. Keep the units.
  - **`date_shift`:** shift by ±{7 days, 1 month, 1 year}, or move to another event's date. The result must fall outside every date range the corpus gives for that fact.
  - **`negation`:** flip the polarity of the relation using a curated relation→antonym table (e.g. "evacuated" ↔ "not evacuated"; "disengaged from" ↔ "did not disengage from").
  - **`misattribution`:** swap `attributed_source` for another source in the scenario, e.g. a statement attributed to MEA becomes attributed to the MFA. Only when the corpus records who said it.
  - **`out_of_context`:** attach a real fact from a distractor event (another year or flood season) to this scenario's event and date.
  - **`fabricated_event`:** combine scenario entities with an event template that appears nowhere in the sources (checked by BM25 over all sources). Label: NEI-fabricated.
- Every output records `rule_id, rule_version, original_slot, new_slot, label`, and a **contradicting `sent_id`** (for FALSE).
- **Tests:**
  - Property tests: every FALSE item differs from its source tuple in exactly one slot.
  - The perturbed value never appears for that fact anywhere in the corpus.
  - The output is deterministic for a given seed.
- **Done when:** `claims_struct.parquet` exists (structured claims plus labels, before rewording), `pytest tests/synth/` passes, and the counts per type meet the targets with margin (generate about 2× the targets, to allow for filtering).

### M16.5 — Surface realisation (LLM rewording only)
- **Action:** `src/ares/synth/realise.py`.
  - **Model:** Mistral-7B-Instruct-v0.3 in 4-bit (NF4 or GGUF Q4_K_M). Record the revision hash.
  - **Prompt:** fixed, in `configs/m16/realise_prompt.txt`. It tells the model to *express exactly these slot values, add no facts, use neutral, non-inflammatory wording*.
  - **Registers (4):** news headline, social-media post, forwarded-message style, official-statement style.
  - **Decoding:** temperature 0.7, top-p 0.9, with a seed stored per item. Make k = 2 realisations per structured claim.
  - **Same treatment for all labels:** TRUE, FALSE and NEI claims go through the **same** prompt and registers, so that style can't reveal the label.
  - **Template control:** 10 % of the structured claims are also realised by **templates only** (no LLM), as a control subset.
  - Cache every prompt, output, seed and model hash in `generations.parquet`. Later rebuilds use this cache, so the benchmark is exactly reproducible.
- **Done when:** the generations are cached with full provenance, and generation GPU-h and peak VRAM are logged.

### M16.6 — Automatic validation
- **Action:** `src/ares/synth/validate.py` runs these checks in order. Every rejection is logged with its reason in `rejections.parquet`.
  1. **Slot preservation:** re-extract the slots from the generated text (the M16.3 extractor plus normalisation: numbers, dates, GeoNames alias lists). Every slot must match the structured claim, and no extra numbers, dates or places may appear.
  2. **Validator NLI:** use **two** NLI models, neither of which is the M04 N-operator model, and neither trained on FEVER. Record both model IDs. Example: `roberta-large-mnli` plus BART-large-MNLI, if M04 chose a DeBERTa-v3 checkpoint. Apply the PREREG thresholds for TRUE, FALSE and NEI.
  3. **Deduplication:** MinHash (Jaccard ≥ 0.8) and embedding cosine ≥ 0.95 between items of **different** facts; keep one. Items of the same fact are kept but share a `fact_id` cluster.
  4. **Safety and style:**
     - a toxicity classifier (e.g. Detoxify) with a pre-registered threshold;
     - a lexicon of slurs and incitement terms;
     - every PERSON entity must be on the public-figure allow-list;
     - no imperative calls to action.
- **NLI-selection bias control:** NLI filtering can favour claims that are easy for NLI-based systems. So also keep the **"NLI-borderline pool"**: items rejected *only* by the NLI threshold while passing all the code checks. This pool is human-labelled in M16.9, and in M12.6b ARES is reported on it separately.
- **Output:** `tables/validation_pass_rates.csv` (per scenario × label × type × register).
- **Done when:** pass rates are reported, the borderline pool is saved, and the kept set meets the targets (or a shortfall is reported).

### M16.7 — Shortcut / artifact check (diagnostic models only; discarded afterwards)
- **Action:** `scripts/m16/artifact_check.py`. These diagnostic models are trained on ARES-Synth itself and **never** enter ARES.
  - 5-fold CV, grouped by `fact_id`, for:
    - TF-IDF+LR (word + char);
    - DeBERTa-v3-base, claim-only, K = 3;
    - a feature model (length, digits, hedges, punctuation, register).
  - **If any of them exceeds the ceiling** (macro-F1 > 0.60, T vs F): run AFLite-style adversarial filtering. Repeatedly train an ensemble of these claim-only models on random 50 % splits, and remove the items they predict correctly at least 75 % of the time, until the ceiling holds or 25 % of the items have been removed.
  - **Do not** filter on any evaluated ARES model (for example the Qwen member). That would bias the benchmark against it.
  - Report before/after numbers and the label and type balance after filtering.
- **Done when:** `results/artifact_check.json` shows the ceiling holds, or the limitation is documented. In that case, the claim-only baseline must sit beside every ARES-Synth result.

### M16.8 — Mutation chains (for M06)
- **Action:** `src/ares/synth/chains.py`. For each sampled TRUE fact:
  - Build a chain `c_0 … c_L` (L ∈ [6, 10]) of realisations.
  - **Benign steps** reword the *same* tuple.
  - **Mutation points** (0, 1 or 2, at pre-drawn positions) apply an M16.4 perturbation; the steps after a mutation reword the *mutated* tuple.
  - Add synthetic timestamps (seeded inter-arrival times) for drift timelines.
  - Every step goes through the M16.6 checks.
  - Record `chain_id, step, tuple_hash, is_mutation, mutation_type, timestamp`.
- **Done when:** `chains.parquet` exists with ≥ 300 chains per scenario, plus a balance table.

### M16.9 — Human audit (HUMAN-GATED; write `BLOCKED.md` when ready)
- **Action:**
  - Build `scripts/m16/audit_ui.py`, a local static HTML or Streamlit form. Each screen shows one claim plus the top-3 evidence sentences (or "no evidence found", for NEI items). It hides the construction label, the perturbation type and the register.
  - **Sample:** n = 300 from the kept set, stratified by scenario × label × type; plus 100 from the NLI-borderline pool; plus all CONTESTED items, up to 50. Seed `20261013`; the order is randomised.
  - **The human records:** T / F / NEI / Contested, "unnatural wording" (y/n), "unsafe or inflammatory" (y/n), and an optional comment.
  - **Annotator independence:** the primary annotator is the author, which is a conflict of interest. Mitigate it with blinding and randomisation, and **strongly recommend a second, independent annotator** on at least 150 items. Report Cohen's κ between annotators if there is a second one, and κ between each annotator and the construction labels.
- **Metrics:**
  - accuracy against construction labels, with a Wilson CI;
  - per-type accuracy;
  - the rate of naturalness and safety flags;
  - borderline-pool labels, which become their gold labels for M12.6b;
  - agreement on the CONTESTED items.
- **Decision:** if the PREREG thresholds fail, diagnose the failing type(s), fix them, and rebuild the affected items (iteration v1a; at most 2 iterations). Re-audit a fresh sample.
- **Done when:** `results/human_audit.json` and `tables/human_audit.csv` exist, and the acceptance decision is recorded.

### M16.10 — Splits & freeze
- **Action:**
  - The whole of ARES-Synth is **test**, except a 100-item `smoke` subset, which is used only to check that the M12 pipeline runs and is excluded from every reported number.
  - Write `ares_v2/data_manifests/ares_synth_v1.csv`: `item_id, scenario, label, type, register, fact_id, chain_id, source sent_ids, contradicting sent_id, gold GeoNames IDs, realisation (llm|template), sha256`. Also write its manifest hash.
  - Commit, and tag `ares-synth-v1`.
  - Add a "frozen" row to `milestones/test_set_access_log.csv`.
- **Done when:** the tag exists, and the hash is recorded in this milestone's README and in `03_IMPLEMENTATION_PLAN.md` §6 notes.

### M16.11 — Datasheet, data statement, licence
- **Action:** `milestones/M16_synthetic_stress/DATASHEET.md` (Gebru et al. format) covering:
  - motivation;
  - the construction pipeline (with a diagram);
  - sources and licences: Wikipedia CC BY-SA → the derived claims are CC BY-SA; ReliefWeb terms; government statements;
  - the label semantics, including the CONTESTED policy;
  - generator and validator model IDs;
  - known biases (NLI selection, English-only, template/LLM style, LLM pre-training contamination for 2020/2022 events);
  - intended use: **evaluation of verification systems only; not a source of facts; not for training propaganda detectors to be deployed without real data**;
  - prohibited uses;
  - the SIMULATED label requirement;
  - maintenance.
- **Release rule:** release only items derived from openly licensed sources. News-derived evidence is released as URL + hash.

### M16.12 — Ledger delta
- **Action:** `ledger_delta.csv` sets `target_milestone=M16/M12` for C020-India, C021 and C022, with the note "benchmark built and frozen (`ares-synth-v1`); measured in M12.6b; final status REVISED with the qualifier 'on a controlled grounded-synthetic benchmark'". No measured values here, because nothing has been evaluated yet.

---

## Audit gate — milestone-specific red flags
- RF1: **Labels from code.** For a random 50 items, the auditor re-derives the label from `claims_struct.parquet` plus the perturbation log. Any label that only an LLM produced is a BLOCKER.
- RF2: **No ARES contact before the freeze.** No log, results JSON or cache shows any ARES module (including the M04 N model and the Qwen member) run on ARES-Synth before the `ares-synth-v1` tag date.
- RF3: **Family and model separation.** The generator is not Qwen-family; neither validator NLI is the M04 N model; the M16.3 extractor is not the M04 logic-layer parser.
- RF4: **Withheld sentences** are really absent from the evidence corpus (exact match and near-duplicates).
- RF5: **Contested facts** never appear as TRUE or FALSE (the auditor greps the contested list against kept items).
- RF6: **Shortcut ceiling** holds, or the limitation is documented. Diagnostic models were not reused anywhere.
- RF7: **Slot preservation:** 30 random LLM realisations are checked by hand for an added or dropped fact.
- RF8: **Safety:** no private individuals; toxicity and incitement checks pass; a manual sample of 50 is checked.
- RF9: **Clustering:** `fact_id` is present for every item, and the CV and CI procedures group by it.
- RF10: **Reproducibility:** rebuilding from the cached generations reproduces the manifest hash exactly.
- RF11: **SIMULATED labelling** appears in every file name, JSON, table and README section.

## Exit criteria
- [ ] Scenario cards, frozen evidence corpus, facts, perturbations, realisations, validation, artifact check and chains all complete, with counts meeting the targets (or the shortfall documented).
- [ ] Human audit done and passing, or the iteration limit reached and documented.
- [ ] Datasheet and licence complete. Manifest committed. Tag `ares-synth-v1`. Freeze logged.
- [ ] `ledger_delta.csv` written.
- [ ] AUDIT.md: no open BLOCKER/MAJOR. Tag `m16-complete`. **Must happen before the first M11 run.**
