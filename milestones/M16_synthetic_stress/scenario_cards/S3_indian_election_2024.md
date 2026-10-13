# Scenario Card S3: Indian General Election 2024 (Electoral Claims & Procedural Certifications)

**Scenario Identifier:** `S3`  
**Status:** `SIMULATED — grounded synthetic complement`  
**Ledger Target:** `C020` (Election manipulation F1 / Indian disaggregated component)  
**Temporal Window:** 2024-03-16 → 2024-06-10  

---

## 1. Context & Scope

This scenario models verification of procedural, logistical, and certified statistical claims during the 18th Lok Sabha general election in India. Key components include:
1. The seven-phase election schedule announced on 16 March 2024.
2. Official constituency counting and declaration of results for all 543 elected parliamentary seats on 4 June 2024.
3. Official certification of total voter turnout (65.79%) across phases.
4. Procedural safeguards, EVM-VVPAT tally verifications, and ECI statutory notifications.

---

## 2. Primary Grounded Sources

| Source Entity | Document / Bulletin Title | Temporal Anchor | Manifest / Snapshot Ref |
|:---|:---|:---|:---|
| **Election Commission of India (ECI)** | Press Note No. ECI/PN/23/2024: Schedule for General Election to Lok Sabha 2024 | 2024-03-16 | `M02.10b_eci_schedule_20240316` |
| **Election Commission of India (ECI)** | Official Constituency-Wise Result Declarations & Trends Portal | 2024-06-04 | `M02.10b_eci_results_20240604` |
| **Election Commission of India (ECI)** | Press Release: Final Voter Turnout Recorded in General Election 2024 | 2024-06-06 | `M02.10b_eci_turnout_20240606` |
| **Supreme Court of India** | Judgment in *Association for Democratic Reforms v. ECI* (EVM-VVPAT Verification) | 2024-04-26 | `M02.10b_sci_vvpat_20240426` |
| **Press Information Bureau (PIB)** | Daily Fact Check Dispatches on Electoral Misinformation | 2024-04-01 → 2024-06-05 | `M02.10b_pib_elections_2024` |

---

## 3. Entity Registry (Wikidata QIDs)

| Entity Name | Entity Category | Wikidata QID | Operational Role |
|:---|:---|:---|:---|
| **Election Commission of India (ECI)** | Constitutional Body | `Q5354395` | Conduct of parliamentary and assembly elections |
| **Lok Sabha** | Lower House of Parliament | `Q213346` | Target legislative assembly (543 seats) |
| **National Democratic Alliance (NDA)** | Political Coalition | `Q1371510` | Governing electoral coalition |
| **Indian National Developmental Inclusive Alliance (INDIA)** | Political Coalition | `Q120733355` | Opposition electoral coalition |
| **Voter Verifiable Paper Audit Trail (VVPAT)** | Auditing Device | `Q17071661` | Independent EVM voter verification slips |
| **Electronic Voting Machine (EVM)** | Voting System | `Q5358327` | Tamper-evident standalone voting apparatus |

---

## 4. Geographic Gazetteer (GeoNames IDs & Coordinates)

| Constituency / Centre | Administrative Level | GeoNames ID | Latitude | Longitude |
|:---|:---|:---|:---|:---|
| **New Delhi** | National Capital / Nirvachan Sadan HQ | `1261481` | 28.6100° N | 77.2000° E |
| **Varanasi** | Parliamentary Constituency | `1253405` | 25.3200° N | 82.9700° E |
| **Gandhinagar** | Parliamentary Constituency | `1271157` | 23.2200° N | 72.6800° E |
| **Wayanad** | Parliamentary Constituency | `1253164` | 11.6800° N | 76.1300° E |
| **Kolkata** | State Capital (West Bengal) | `1275004` | 22.5700° N | 88.3600° E |

---

## 5. Perturbation & Grounded Realism

- **Constituency Counting:** Certified tally is 543 elected parliamentary seats (Anglo-Indian nominations having been abolished by the 104th Constitutional Amendment). Numerical perturbations alter this number (e.g. to 545 or 542).
- **Voter Turnout:** The aggregate certified turnout across the 7 phases is 65.79%. Number perturbations alter this percentage (e.g. to 67.4% or 58.2%), producing unambiguous factual contradictions with official ECI press notices.
- **Phasing & Scheduling:** Perturbations test confusion across the 7 polling phases (19 April to 1 June 2024).

---

## 6. Public Figure Allow-List

Only constitutional dignitaries and election commissioners:
- **Rajiv Kumar** (`Q112028682`) — Chief Election Commissioner of India
- **Gyanesh Kumar** — Election Commissioner of India
- **Dr. Sukhbir Singh Sandhu** — Election Commissioner of India
- **Narendra Modi** (`Q1058`) — Prime Minister of India (Candidate, Varanasi)
- **Rahul Gandhi** (`Q253859`) — Member of Parliament (Candidate, Wayanad / Rae Bareli)
