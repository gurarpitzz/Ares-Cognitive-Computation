# Scenario Card S1: India–China Border Conflict (Eastern Ladakh & Tawang)

**Scenario Identifier:** `S1`  
**Status:** `SIMULATED — grounded synthetic complement`  
**Ledger Target:** `C021` (Border conflict propaganda recall / disaggregated evaluation)  
**Temporal Window:** 2020-05-01 → 2023-01-31  

---

## 1. Context & Scope

This scenario models information flow and verification during military standoffs and disengagement negotiations along the Line of Actual Control (LAC). It encompasses:
1. The June 2020 Galwan Valley clash.
2. The February 2021 disengagement along the North and South banks of Pangong Tso.
3. The December 2022 skirmish in the Yangtse sector near Tawang, Arunachal Pradesh.
4. The October 2024 diplomatic agreements on patrolling arrangements in Eastern Ladakh.

---

## 2. Primary Grounded Sources

| Source Entity | Document / Bulletin Title | Temporal Anchor | Manifest / Snapshot Ref |
|:---|:---|:---|:---|
| **Ministry of External Affairs (MEA), India** | Official Statement on violent face-off in Galwan Valley | 2020-06-16 | `M02.10b_mea_galwan_20200616` |
| **Ministry of Defence (MoD), India** | Statement by Raksha Mantri Rajnath Singh in Rajya Sabha on Pangong Tso Disengagement | 2021-02-11 | `M02.10b_mod_pangong_20210211` |
| **Lok Sabha Secretariat** | Suo Motu Statement by Raksha Mantri on Yangtse/Tawang Incident | 2022-12-13 | `M02.10b_parl_tawang_20221213` |
| **Ministry of External Affairs (MEA), India** | Press Briefing by Foreign Secretary Vikram Misri on LAC Patrolling | 2024-10-21 | `M02.10b_mea_patrol_20241021` |
| **Ministry of Foreign Affairs (MFA), PRC** | Regular Press Conference Spokesperson Statements | 2020-06-19 | `M02.10b_prc_mfa_20200619` |

---

## 3. Entity Registry (Wikidata QIDs)

| Entity Name | Entity Category | Wikidata QID | Operational Role |
|:---|:---|:---|:---|
| **Indian Army** | Armed Forces | `Q162985` | Ground defence and border patrolling |
| **People's Liberation Army (PLA)** | Armed Forces | `Q200106` | Ground forces along LAC |
| **Ministry of External Affairs (India)** | Ministry | `Q3534577` | Official diplomatic communications |
| **Ministry of Foreign Affairs (PRC)** | Ministry | `Q1156108` | Official Chinese foreign affairs dispatches |
| **Line of Actual Control (LAC)** | Border / Demarcation | `Q1417066` | Disputed demarcation line |
| **Indo-Tibetan Border Police (ITBP)** | Border Police | `Q1661643` | Border guarding organization |
| **Border Roads Organisation (BRO)** | Infrastructure Agency | `Q3348636` | Strategic infrastructure construction |

---

## 4. Geographic Gazetteer (GeoNames IDs & Coordinates)

| Location Name | Region / Sector | GeoNames ID | Latitude | Longitude |
|:---|:---|:---|:---|:---|
| **Galwan Valley** | Eastern Ladakh | `1271439` | 34.7600° N | 78.2100° E |
| **Pangong Tso** | Eastern Ladakh | `1260203` | 33.7200° N | 78.9100° E |
| **Hot Springs (PP15)** | Eastern Ladakh | `1269550` | 34.2500° N | 78.9700° E |
| **Tawang (Yangtse Sector)** | Arunachal Pradesh | `1254832` | 27.5800° N | 91.8600° E |
| **Leh** | Ladakh UT HQ | `1266938` | 34.1500° N | 77.5800° E |

---

## 5. Contested Facts Policy & Disputed Claims

Any claim touching upon the following items is classified as **`CONTESTED`** and **strictly excluded** from binary and 3-way evaluation metrics:
1. **PLA Casualty Figures at Galwan (June 2020):**
   - *Chinese Official Account:* 4 soldiers posthumously awarded heroic honors in February 2021 (PLA Daily / Xinhua).
   - *Indian & Independent Intelligence Estimates:* Claimed between 35 and 45 PLA casualties (TASS, US Congressional Commission reports).
   - *Policy:* Contested. Ideal system output is epistemic abstention ("Uncertain / Inconclusive").
2. **Initiation of Physical Contact:**
   - *Indian Account:* PLA erected tents on the Indian side of LAC and breached disengagement consensus.
   - *Chinese MFA Account:* Indian frontline troops crossed LAC into Chinese territory to provoke PLA officers.
   - *Policy:* Contested.
3. **Sovereignty Assertions:**
   - Contested statements regarding sovereign claims over the Galwan river bend or Fingers 4–8 on Pangong Tso.

---

## 6. Public Figure Allow-List

Only official public figures named in official press briefings are permitted; private individuals or deceased soldiers' family names are strictly forbidden:
- **Rajnath Singh** (`Q1435275`) — Raksha Mantri (Minister of Defence, India)
- **Dr. S. Jaishankar** (`Q15116827`) — External Affairs Minister, India
- **Vikram Misri** (`Q28181673`) — Foreign Secretary, India
- **Zhao Lijian** (`Q66723223`) — Spokesperson, MFA China
- **Wang Wenbin** (`Q96312458`) — Spokesperson, MFA China
