# ARES — Manuscript Integrity Audit (Read This First)

**Audited:** `Ares__Cognitive_computation.pdf` (39 pp), `ARES_response_to_reviewers.pdf`, `reviewer_comments.txt`, and the public repo `github.com/gurarpitzz/ARES_POC` (state as of 2026-10-13).
**Purpose:** Before any code is written, establish exactly what exists, what does not, and which manuscript statements are internally inconsistent, mathematically wrong, or impossible to reproduce with code. Every implementation milestone takes this document as input.

---

## 0. Bottom line

1. **Only one component of ARES actually exists as code:** the evidence functional
   `Truth'(C) = Σ S·N·W / Σ S`, implemented in `ARES_POC` as: web search → sentence split → MiniLM cosine Top‑M → BART‑MNLI label (+1/0/−1) → domain-keyword credibility weight. That's about 30 KB of Python. There is no training code, no trained model, no deepfake/audio/bot/geo/drift module, no fusion head, no ablation code, no statistical tests, no latency harness.
2. **Every quantitative result in Sections X–XXX has no code behind it**: F1 = 0.92, 415 ms, every ablation, every stress test, the significance tests, the HITL numbers, the latency-vs-token table. This also covers results the response letter says were added (K = 5 runs, t‑test, Wilcoxon, Cohen's d, ablations).
3. **Some reported numbers are internally inconsistent or mathematically impossible** (§2–§3 below). They can't have come from a real experiment as described, so they can't be "regenerated". They have to be replaced with whatever the new code measures.
4. **Some claims describe real-world events and human studies** (military alerts, 13,000 posts removed, a −47 % traffic reduction, a survey with n = 4000, analyst response times). Code can't produce these. They must be removed, or turned into clearly labelled hypothetical scenarios, unless a real study is carried out.
5. **The idea is good, and a lot can be rescued.** Reviewer 3 called the concept innovative, and that holds up. The evidence functional, the decision-theoretic verifier, and the drift/mutation model can all be built, measured, and (for the theory) properly proven. §6 lists what can be salvaged.

> **Governing principle for the rebuild:** We build the system the paper describes, run pre-registered experiments on public data, and **report whatever the experiments produce.** The manuscript gets rewritten from the measured results. Code is never written or tuned to hit a number that's already in the paper. If a measured number comes out lower than the paper's, the paper changes, not the experiment.

If this manuscript version is currently under review anywhere, the unsupported numbers should not stay in front of reviewers. Contact the editor, withdraw, or update before the new results exist.

---

## 1. What exists vs. what the paper claims

| Manuscript component | Section | Exists in code? | Notes |
|---|---|---|---|
| Evidence functional R·S·N·W | XXXI | **Partial** | Live web search (not reproducible); no frozen corpus; no evaluation beyond 2-sample fallbacks |
| `evaluate.py` LIAR/FEVER | XXXI-B | **Broken by design** | Quietly falls back to hard-coded 2-claim toy datasets if download fails; uses weighted F1 over 3 classes with "UNCERTAIN" mapping; FEVER path needs local wiki dump that isn't provided |
| Text verifier (DeBERTa-v3-base, 15 epochs, AdamW) | XXV, App. A–B | **No** | |
| Layer-1 ensemble {LLM, DeBERTa, JBC} + weighted voting | VI | **No** | |
| Log-likelihood verifier L(c), τ0/τ1 | XVI, XXI | **No** | |
| Deepfake detector (FF++), ResNet-50 | X, App. A | **No** | |
| Voice tampering / speaker embedding | X | **No** | |
| Lip-sync Δsync | X-C | **No** | |
| Stylometry + OCSVM, HMM, centrality | VII | **No** | |
| Claim mutation / drift tracker | VIII, XVII | **No** | |
| GDELT / Wikidata / Factbook integration | IX | **No** | |
| Fusion head (1024-d, GELU, 3-way softmax) | App. A | **No** | |
| Ablations, cross-domain, LOSO, stress tests | XXII–XXX | **No** | |
| Significance tests, K = 5 seeds | XXV | **No** | |
| Latency/throughput/complexity measurements | XV, XX | **No** | |
| HITL escalation statistics | XXVIII | **No** | |
| Dashboard, heatmap, alerts, audit log, PII | XI, XIV | **Minimal UI only** | `index.html` + FastAPI for the POC trace |

---

## 2. Internal inconsistencies (numbers that contradict each other inside the paper)

| # | Inconsistency | Where | Detail |
|---|---|---|---|
| I‑1 | Same ablation, three different values (Geo) | Tab. XXIII vs XXVIII vs §XXVII/Tab. XXXII | −Geo F1 drop = **0.07** / **0.09** / **0.043** |
| I‑2 | Same ablation, three values (Behavior) | Tab. XXIII vs XXIX vs XXXII | **0.08** / **0.06** / **0.031** |
| I‑3 | Same ablation, two values (Temporal) | Tab. XXX vs XXXII | **0.08** / **0.049** |
| I‑4 | Multi-modal removal | Tab. XXIII vs §XXVII | −Vision (0.06) + −Audio (0.04) and Text-only (−0.13) vs. ΔF1_multimodal = **0.071** |
| I‑5 | "Super-additive" interaction claim is false by the paper's own numbers | §XXVII-C, Eq. 37, Eq. 59 | 0.112 **<** 0.071 + 0.049 = 0.120 → **sub-additive** |
| I‑6 | Contribution shares | §XXVII-E | From the paper's own Δs: drift = 0.194, MM = 0.282 (paper says 0.21, 0.30) |
| I‑7 | Baseline table duplicated with different numbers | Tab. XXI vs XXXVII | ClaimBuster latency 640 vs 540; FakeFlow F1 0.84 vs 0.83, latency 580 vs 680; Grover F1 0.87 vs 0.86, latency 712 vs 720 |
| I‑8 | Latency exponent contradicts own data | Tab. XXII vs §XX-C | Regression on Table XXII gives slope **1.44** (all points) / **1.36** (64–512). Paper reports **α = 1.99 ± 0.04** |
| I‑9 | HITL expected latency arithmetic | Eq. 47–48 | (1−0.068)·0.415 s + 0.068·18.6 min = **76.3 s**, not 1.26 s |
| I‑10 | Significance tests cite the wrong table | §XXV-C | "benchmark comparisons reported in Table XIII". Table XIII is a latency table |
| I‑11 | Output space | §XXV-A vs App. A-d | Training is binary y ∈ {0,1} with BCE; the architecture outputs a 3-way softmax |
| I‑12 | Four different decision rules | §III-E, §XVI-C, §XXXI, Alg. 2 | F(e) ≥ τ; L > τ1; Truth' < −0.4; FinalScore ≥ 0.75 / ≤ 0.45. Never unified |
| I‑13 | Language support | Abstract/Tab. XXI vs App. A | "14 languages", but the declared text encoder (DeBERTa‑v3‑base) is English-only |
| I‑14 | Table XI F1 vs TPR/FPR | Tab. XI | With balanced classes, TPR/FPR imply F1 = 0.962 / 0.970 / 0.904 / 0.851 (reported 0.97 / 0.98 / 0.92 / 0.88). Possible under class imbalance, but the class ratio is never stated |
| I‑15 | Fig. 25 axis | §XIX | Density of a cosine similarity plotted on x ∈ [−6, 6]; cosine is bounded in [−1, 1] |
| I‑16 | Double regularisation | Eq. 18 + AdamW wd | λ_reg‖θ‖² **and** AdamW weight decay 0.01. Not wrong, but undisclosed redundancy |

## 3. Mathematically impossible or incorrect statements

| # | Statement | Why it's wrong | Fix |
|---|---|---|---|
| M‑1 | "Wilcoxon signed-rank … p < 0.01" with K = 5 runs | The smallest achievable two-sided exact Wilcoxon p with 5 pairs is **0.0625** (verified). p < 0.01 can't happen | Use K ≥ 10 seeds (min p = 0.00195) **and** instance-level paired bootstrap / McNemar |
| M‑2 | Cohen's d ∈ [0.8, 1.3] **and** paired t p < 0.01 at K = 5 | t = d·√K ∈ [1.79, 2.91], so p ∈ [0.044, 0.148]. p < 0.01 needs d > 2.06 | Report the actual d, t, p |
| M‑3 | RoBERTa latency measured at n = 768, 1024 tokens | RoBERTa has 512 position embeddings and can't take 768/1024 tokens without modification | Measure ≤ 512 on RoBERTa/DeBERTa; use a long-context encoder (e.g. ModernBERT/Longformer) for longer n and say so |
| M‑4 | "For typical n < d, the quadratic attention term dominates" | For n < d, n²d < nd², so the **FFN/projection** term dominates. Attention dominates only when n > d (roughly) | Correct the analysis; this also explains why the measured slope is < 2 |
| M‑5 | E[cos(x_t, x_t+ε)] ≈ 1 − Tr(Σ)/(2‖x‖²) | Drops the component of ε parallel to x. Correct second-order form: 1 − (Tr(Σ) − x̂ᵀΣx̂)/(2‖x‖²) | See M01 (exact result available) |
| M‑6 | §XXI-F: Δt ≈ 1 − ½‖ε‖² | Missing normalisation by ‖x‖² and the parallel-component correction | As above |
| M‑7 | Proposition 1 "detected with probability ≥ 1−δ for any δ" | As stated, it's vacuous: θ → 1 flags *everything*, so TPR → 1 trivially. A detection guarantee has to bound **both** miss rate under H1 and false-alarm rate under H0. The Lipschitz assumption (1) is never used in the proof. The "Chernoff-type" step isn't derived | Restate as a two-sided detection theorem (M01) |
| M‑8 | "decision-theoretically optimal classifier", "classical Neyman–Pearson test" | The LLR is NP-optimal only for the *true* likelihoods of a simple binary test. With learned weights w_i it's a logistic-regression score; a three-way rule with abstention is Chow's rule, not NP. Optimality requires calibrated likelihoods and (for the sum) conditional independence of sources | State the assumptions; show calibration empirically; derive τ0, τ1 from explicit costs (M05) |
| M‑9 | Eq. 29 bound + "ARES explicitly minimizes disc" | Nothing in the training objective minimises a discrepancy term. The bound can be *estimated* (proxy A-distance) but isn't optimised | Estimate the proxy A-distance between domains and correlate it with the measured drop (M12) |
| M‑10 | Conf(C) = |Truth'|·log(1+M) called "confidence" | Unbounded: with M = 7, values up to 2.08. Not a probability | Replace with a calibrated probability (M05) |
| M‑11 | CI95 = F̄1 ± 1.96·σ/√n described as "bootstrap" | That's a normal approximation, not a bootstrap CI | Percentile/BCa bootstrap (M00 statistical protocol) |
| M‑12 | Eq. 62: R_Dr(f) ∝ 1/|Dr| | Not a known result; standard rates are O(√(complexity/|D|)) | Remove or replace with a cited bound |

## 4. Claims that code cannot reproduce (remove, reframe, or run a real study)

| Claim | Section | Required action |
|---|---|---|
| Level‑3 alert within 18 min; panic traffic −47 %; heatmap integrated into national dashboard | XIII-A | **Remove.** Re-present as a hypothetical scenario with no impact numbers, or as a retrospective case study on real historical claims with real system traces |
| "Classified alert issued to military intelligence, preventing escalation" | XIII-B | **Remove** |
| Verdict confidence 94.7 %; PSA campaign 6 h earlier | XIII-C | **Remove** impact claim; the confidence can be shown from a real trace |
| 13,000+ bot-amplified posts removed | XIII-D | **Remove** |
| +62 % trust gain (survey n = 4000) | XIII-E | **Remove** unless an ethics-approved user study is actually run |
| Analyst response time by severity (Fig. 22); T_human = 18.6 min | XIII, XXVIII | **Remove** or run a real (small, consented) analyst study. Can be replaced by a parameterised sensitivity analysis |
| Live OSINT (Telegram, regional X) agreement > 88 % with IFCN | XXVI-E | **Replace** with a real post-freeze temporal validation on ClaimReview fact-checks (M12) |
| Crowdsourced labelling via MTurk + IFCN consortium | XII-B | **Remove** or describe as proposed design, not performed work |
| India–China border propaganda recall 91.1 %; Assam flood 2022 accuracy 94.8 % | XII-D | No public labelled dataset is known to me. The real-world numbers are **removed**. The scenarios are re-tested on **ARES-Synth** (M16): a grounded synthetic benchmark built from frozen real sources, with code-determined labels, independent validation, shortcut checks and a human audit. It is evaluated test-only in M12.6b. The claims become **REVISED** with the qualifier "on a controlled grounded-synthetic benchmark" and are always labelled SIMULATED. A real labelled subset, if M12.6 finds one, takes precedence |
| Federated learning, blockchain audit log, ethical licensing, kill switch | XIV | Hash-chained audit log and PII sanitisation can be implemented (M14). Federated learning and governance must be worded as **design proposals**, not implemented features |

## 5. Dataset & baseline problems

- **VoxCeleb is a speaker-identity dataset, not a voice-manipulation dataset** (Tab. XIV, XII-A). Spoofing/tampering detection needs ASVspoof 2019 LA / 2021 (and optionally FakeAVCeleb, In-the-Wild). VoxCeleb is still right for speaker-embedding similarity.
- **"DeFacto-X"** (Tab. XX): I couldn't identify this dataset. Cite it precisely or remove it. **X-FACT** has 25 languages, not 14.
- **ClaimBuster** scores *check-worthiness*, not veracity. Comparing its F1 to a veracity F1 is a category error. **Grover**'s discriminator detects *machine-generated* news, also not veracity. They can only be compared on a task they were built for, or the comparison has to be dropped. Honest veracity baselines: TF-IDF+LR, fine-tuned BERT/RoBERTa/DeBERTa, FakeFlow (re-run from its public code), retrieval+NLI pipelines, and an open LLM zero-shot.
- **Seeded paired tests against ClaimBuster/Grover/FakeFlow are impossible** unless those systems are retrained under the same seeds. Pre-trained third-party systems contribute one deterministic run.
- **No single public dataset contains text + image + audio + video + propagation + geo for the same items.** So a single "Full ARES F1 = 0.92" with "−Audio" and "−Vision" ablations on one test set can't exist. The rebuild uses **tracks** (M11), each with the modalities that actually co-occur, and ablations are only reported where the removed modality is present.
- **Evidence leakage risk:** web retrieval for LIAR/PolitiFact claims will often return the PolitiFact verdict page itself. Fact-check domains and post-claim-date evidence have to be excluded (M04).
- **Expected realistic ranges (for leakage detection only, never targets):** LIAR 6-way accuracy ≈ 0.25–0.30 text-only (Wang 2017 reports ≈ 0.27); binary LIAR ≈ 0.60–0.70. An F1 near 0.92 on LIAR would almost certainly mean leakage. FF++ (c23) in-domain AUC is typically very high (≥ 0.95), while cross-dataset Celeb-DF AUC is usually much lower (≈ 0.65–0.80). Strong ASVspoof 2019 LA systems (e.g. AASIST) report EER below 1 %. Check these against current literature in M02.
- **Hardware statement (C033, C034):** the rebuild runs entirely on an HP OMEN 16: RTX 4060 Laptop GPU (8 GB), i7-13620H, 16 GB RAM, under WSL2. All latency and throughput figures will refer to that machine, so the manuscript's A100 / Xeon / 64 GB statement must change. See `03_IMPLEMENTATION_PLAN.md` §5.
- **Data scope:** this is a proof of concept, so all relevant public and form-gated academic datasets are used (plan §4, Tiers 1–2). Tier-2 results are descriptive generalisation evidence.

## 6. References: suspected errors (verify every one via Crossref/Semantic Scholar in M15)

I couldn't match these to real publications as cited, or the authorship looks wrong. Verify before keeping any of them:
[1] Zhou et al., IEEE Access 2021 · [4] Helbing et al., Nat. Hum. Behav. 2018 · [6] "M. Schuster et al." for *Speech Recognition with Deep RNNs* (ICASSP 2013 is Graves, Mohamed, Hinton) · [9] Shu et al., CVPR 2022 · [13] Grinberg et al., *Science* 2020 · [14] Hasanuzzaman et al., ACL 2021 · [22] Zhang et al., ACL 2020 (QA-based factual consistency is Durmus et al. FEQA / Wang et al. QAGS, ACL 2020) · [26] first author is **Gordon** Pennycook · [27] Kim et al., CIKM 2018 · [35] Tschannen et al., NeurIPS 2020 · [40] Nguyen et al., ACL 2023 · [41] Silva & Ribeiro, JAIHC 2021 · [47] DFDC Preview is Dolhansky et al. 2019 · [49] that title is Yuan et al., IEEE TNNLS 2019. Also: [16] (WSJ CSR corpus) and [8] (YOLOv3) don't appear to support any claim in the text.

## 7. What is genuinely strong and worth building

1. **The evidence functional** Truth'(C) is transparent and auditable. Rebuilt on frozen corpora (FEVER wiki, AVeriTeC knowledge store), it becomes a reproducible, evaluable verifier with a per-evidence trace. This is the paper's most defensible artefact.
2. **The decision-theoretic verifier** becomes rigorous once framed honestly: per-source calibrated log-odds → learned weights (logistic regression = weighted LLR) → Chow-style abstention with explicit costs → risk–coverage curves. That's a real, testable contribution.
3. **The drift/mutation model can be made a real theorem.** Under isotropic Gaussian mutation the stability statistic has an *exact* distribution: cos(x, x+ε) = T/√(T²+d−1) with T ~ noncentral‑t(ν = d−1, λ = ‖x‖/s) (Monte-Carlo-verified in M01). That gives exact ROC curves and Neyman–Pearson thresholds. Then **test the assumptions on real embeddings** (they're probably anisotropic, and the theory must then handle general Σ). Falsifying an assumption honestly *is* rigour.
4. **Claim memory with mutation-aware matching** (retrieving previously fact-checked claims, with strict temporal ordering) gives the drift module a concrete, measurable path to veracity performance, so the "temporal ablation" becomes meaningful.
5. **GDELT event corroboration** (does an event of the claimed type exist at the claimed place/date?) gives the geo module a concrete, measurable contribution for event claims like the J&K example.
6. **Integration as the contribution** (§XXIX) is the right framing. Per-track results with honest ablations and Shapley attributions will make that case much better than one headline F1.
