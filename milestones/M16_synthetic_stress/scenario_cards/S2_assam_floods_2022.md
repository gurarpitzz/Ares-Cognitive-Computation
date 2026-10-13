# Scenario Card S2: Assam Floods 2022 (Disaster Misinformation & Emergency Response)

**Scenario Identifier:** `S2`  
**Status:** `SIMULATED — grounded synthetic complement`  
**Ledger Target:** `C022` (Flood misinformation accuracy / disaggregated evaluation)  
**Temporal Window:** 2022-04-01 → 2022-10-31  

---

## 1. Context & Scope

This scenario captures disaster crisis communications during the severe 2022 Assam monsoon floodwaves. Key events include:
1. Submergence of Silchar town following the breach of the Bethukandi dyke along the Barak river in June 2022.
2. Inundation across 34 of 35 administrative districts affecting over 5.4 million residents.
3. Evacuation operations conducted by NDRF, SDRF, and the Indian Army across Cachar, Dima Hasao, and Nagaon.
4. Cumulative fatality milestones (reaching 197 verified deaths due to floods and landslides).

---

## 2. Primary Grounded Sources

| Source Entity | Document / Bulletin Title | Temporal Anchor | Manifest / Snapshot Ref |
|:---|:---|:---|:---|
| **Assam State Disaster Management Authority (ASDMA)** | Daily Flood Situation Reports (DFSR) | 2022-05-15 → 2022-07-31 | `M02.10b_asdma_daily_2022` |
| **National Disaster Response Force (NDRF)** | NDRF Flood Rescue Deployment Summary Bulletins | 2022-06-18 | `M02.10b_ndrf_assam_20220618` |
| **Central Water Commission (CWC), India** | Daily Hydrological Gauge Bulletins (Dhubri, Guwahati, Silchar) | 2022-06-25 | `M02.10b_cwc_brahmaputra_2022` |
| **District Administration, Cachar** | Press Releases on Silchar Bethukandi Dyke Breach & Relief | 2022-06-22 | `M02.10b_cachar_silchar_20220622` |
| **ReliefWeb / UN OCHA** | India: Floods and Landslides - Situation Report No. 2 | 2022-07-08 | `M02.10b_reliefweb_assam_2022` |

---

## 3. Entity Registry (Wikidata QIDs)

| Entity Name | Entity Category | Wikidata QID | Operational Role |
|:---|:---|:---|:---|
| **Assam State Disaster Management Authority (ASDMA)** | Disaster Authority | `Q4808388` | State-level nodal disaster coordination |
| **National Disaster Response Force (NDRF)** | Emergency Service | `Q6972179` | Specialized disaster search and rescue |
| **Central Water Commission (CWC)** | Technical Commission | `Q5061989` | River stage and flood forecasting |
| **Brahmaputra River** | Watercourse | `Q43757` | Major transboundary river |
| **Barak River** | Watercourse | `Q807204` | Major southern Assam drainage basin |
| **Cachar District** | Administrative Division | `Q1025000` | Severely inundated district |
| **Dima Hasao** | Administrative Division | `Q42813` | Hill district affected by railway/landslide damage |
| **Nagaon District** | Administrative Division | `Q42831` | Central Assam flood-impacted district |

---

## 4. Geographic Gazetteer (GeoNames IDs & Coordinates)

| Location Name | Administrative Level | GeoNames ID | Latitude | Longitude |
|:---|:---|:---|:---|:---|
| **Silchar** | Submerged Urban Centre (Cachar) | `1256312` | 24.8300° N | 92.8000° E |
| **Guwahati** | State Capital (Kamrup Metro) | `1271476` | 26.1400° N | 91.7400° E |
| **Nagaon** | District Headquarters | `1262180` | 26.3400° N | 92.6800° E |
| **Haflong** | Hill Station (Dima Hasao HQ) | `1270425` | 25.1800° N | 93.0200° E |
| **Dhubri** | Downstream Gauge Station | `1272782` | 26.0200° N | 89.9800° E |

---

## 5. Perturbation & Verification Nuances

- **Hydrological Precision:** The Brahmaputra at Dhubri reached 0.85 m above danger level on 25 June 2022. Perturbations testing gauge numbers (`number_change`) alter this value by $\ge 25\%$, creating clear contradictions with CWC bulletins.
- **Bethukandi Dyke:** The breach at Bethukandi occurred in the Mahisha Beel area. Perturbations swap the causative dyke or attribute it to artificial sabotage vs hydrological pressure.
- **Relief Numbers:** Tested numbers (e.g. 280,000 residents affected in Silchar; 197 cumulative fatalities) are calibrated against final audited ASDMA reports.

---

## 6. Public Figure Allow-List

Only authorized government leaders and nodal officials:
- **Dr. Himanta Biswa Sarma** (`Q5765063`) — Chief Minister of Assam
- **Gyanendra Dev Tripathi** — Chief Executive Officer, ASDMA
- **Jishnu Barua** — Chief Secretary, Government of Assam (2022)
- **Atul Bora** (`Q4818780`) — Minister for Agriculture & Border Protection
