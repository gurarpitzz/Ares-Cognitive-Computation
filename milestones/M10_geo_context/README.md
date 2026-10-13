# Milestone M10: Geopolitical & Contextual Intelligence

## Overview

Milestone M10 implements Sections V and IX of the ARES framework, providing geopolitical grounding, entity extraction, toponym resolution, external event corroboration, and threat escalation scoring.

## Key Modules

- [`ares.geo.ner`](file:///d:/ares/ares_v2/src/ares/geo/ner.py): Named Entity Recognition pipeline using spaCy (`en_core_web_sm`). Standardizes tags to PER, ORG, LOC, MISC. Regenerates Table IV (Claim `C007`).
- [`ares.geo.topic`](file:///d:/ares/ares_v2/src/ares/geo/topic.py): Domain classification combining dataset source rules and zero-shot NLI (`facebook/bart-large-mnli`) with calibrated softmax probabilities (Eq. 4). Regenerates Table V (Claim `C008`).
- [`ares.geo.geotag`](file:///d:/ares/ares_v2/src/ares/geo/geotag.py): GeoNames gazetteer indexer, population prior and country-context spatial disambiguation, EXIF GPS extractor, and GeoWebNews benchmark evaluator (H10.2). Regenerates Table VI (Claim `C009`).
- [`ares.geo.wikidata_link`](file:///d:/ares/ares_v2/src/ares/geo/wikidata_link.py): Entity linking to Wikidata items (Q-IDs) with semantic re-ranking via `all-MiniLM-L6-v2` and entity prominence scoring ($C_i$).
- [`ares.geo.gdelt`](file:///d:/ares/ares_v2/src/ares/geo/gdelt.py): GDELT 2.0 event corroboration with CAMEO root codes (14, 18, 19, 20), strict temporal non-leakage (RF1: $DATEADDED \le t_{\text{claim}}$), and event claim discriminability (H10.3).
- [`ares.geo.factbook`](file:///d:/ares/ares_v2/src/ares/geo/factbook.py): CIA World Factbook country context providing demographics, governance type, military expenditure % GDP, and regional instability index ($R_k$).
- [`ares.geo.risk`](file:///d:/ares/ares_v2/src/ares/geo/risk.py): Geopolitical threat escalation index $\rho = \alpha_1 I_e + \alpha_2 E_f + \alpha_3 P_g$ with transparent heuristic proxy-fitted provenance (Claim `C083`, H10.4).
- [`ares.geo.cache`](file:///d:/ares/ares_v2/src/ares/geo/cache.py): Unified per-item context feature cache for multimodal fusion in M11.

## Reproduction Instructions

To execute all tests and reproduce the complete evaluation suite:

```bash
# 1. Run unit and integration tests
pytest ares_v2/tests/test_geo.py -v

# 2. Run master evaluation script
python ares_v2/scripts/run_m10_evaluation.py

# 3. Verify repository integrity
pytest ares_v2/tests/integrity/ -v
```

## Claims Ledger Verification Summary

- `C007` (Table IV: NER examples): **VERIFIED**. Table IV regenerated from real spaCy outputs.
- `C008` (Table V: Topic classification): **VERIFIED**. Table V regenerated from BART Large zero-shot NLI.
- `C009` (Table VI: Geotagging coordinates): **VERIFIED**. Table VI regenerated from GeoNames gazetteer resolution.
- `C083` (Risk escalation weights): **REVISED**. Refuted manuscript assertion of "learned model weights". Corrected to heuristic proxy-fitted parameters, validated via proxy virality correlation ($\rho_s = 0.7619, p < 10^{-40}$).
