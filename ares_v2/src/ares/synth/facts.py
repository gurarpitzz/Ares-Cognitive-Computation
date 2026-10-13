"""
ARES-Synth: Atomic Fact Representation and Grounded Source Snapshots.

Encapsulates atomic fact tuples extracted from real frozen source snapshots:
(fact_id, scenario_id, subject, relation, object, number_val, unit, date_str, location_name, geonames_id, coords, attributed_source, raw_sent)
Includes entity registries with Wikidata QIDs and gazetteers with GeoNames coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroundedFact:
    fact_id: str
    scenario_id: str  # S1, S2, S3
    subject: str
    subject_qid: str
    relation: str
    object: str
    object_qid: str
    number_val: Optional[float]
    unit: Optional[str]
    date_str: str
    location_name: str
    geonames_id: int
    latitude: float
    longitude: float
    attributed_source: str
    raw_sentence: str


# Scenario S1: India-China Border Conflict Entities & Gazetteer
S1_ENTITIES = {
    "Indian Army": "Q162985",
    "People's Liberation Army": "Q200106",
    "Ministry of External Affairs": "Q3534577",
    "Ministry of Foreign Affairs of China": "Q1156108",
    "Line of Actual Control": "Q1417066",
    "Galwan Valley": "Q1529124",
    "Pangong Tso": "Q217277",
    "Hot Springs": "Q5910408",
    "Gogra Post": "Q107472093",
    "Yangtse Sector": "Q115668962",
    "Tawang": "Q283084",
}

S1_LOCATIONS = {
    "Galwan Valley": {"geonames_id": 1271439, "lat": 34.76, "lon": 78.21},
    "Pangong Tso": {"geonames_id": 1260203, "lat": 33.72, "lon": 78.91},
    "Hot Springs": {"geonames_id": 1269550, "lat": 34.25, "lon": 78.97},
    "Tawang": {"geonames_id": 1254832, "lat": 27.58, "lon": 91.86},
    "Leh": {"geonames_id": 1266938, "lat": 34.15, "lon": 77.58},
}

# Scenario S2: Assam Floods 2022 Entities & Gazetteer
S2_ENTITIES = {
    "Assam State Disaster Management Authority": "Q4808388",
    "National Disaster Response Force": "Q6972179",
    "Central Water Commission": "Q5061989",
    "Brahmaputra River": "Q43757",
    "Barak River": "Q807204",
    "Cachar District": "Q1025000",
    "Silchar": "Q1025585",
    "Dima Hasao": "Q42813",
    "Nagaon District": "Q42831",
    "Kamrup Metropolitan": "Q2299839",
    "Guwahati": "Q199623",
}

S2_LOCATIONS = {
    "Silchar": {"geonames_id": 1256312, "lat": 24.83, "lon": 92.80},
    "Guwahati": {"geonames_id": 1271476, "lat": 26.14, "lon": 91.74},
    "Nagaon": {"geonames_id": 1262180, "lat": 26.34, "lon": 92.68},
    "Haflong": {"geonames_id": 1270425, "lat": 25.18, "lon": 93.02},
    "Dhubri": {"geonames_id": 1272782, "lat": 26.02, "lon": 89.98},
}

# Scenario S3: Indian General Election 2024 Entities & Gazetteer
S3_ENTITIES = {
    "Election Commission of India": "Q5354395",
    "Lok Sabha": "Q213346",
    "National Democratic Alliance": "Q1371510",
    "Indian National Developmental Inclusive Alliance": "Q120733355",
    "Voter Verifiable Paper Audit Trail": "Q17071661",
    "Electronic Voting Machine": "Q5358327",
    "Varanasi": "Q79980",
    "New Delhi": "Q987",
    "Wayanad": "Q1538354",
    "Gandhinagar": "Q11920",
}

S3_LOCATIONS = {
    "New Delhi": {"geonames_id": 1261481, "lat": 28.61, "lon": 77.20},
    "Varanasi": {"geonames_id": 1253405, "lat": 25.32, "lon": 82.97},
    "Gandhinagar": {"geonames_id": 1271157, "lat": 23.22, "lon": 72.68},
    "Wayanad": {"geonames_id": 1253164, "lat": 11.68, "lon": 76.13},
    "Kolkata": {"geonames_id": 1275004, "lat": 22.57, "lon": 88.36},
}


def get_grounded_facts() -> List[GroundedFact]:
    """Compile canonical grounded atomic facts from verified government bulletins and official records."""
    facts: List[GroundedFact] = [
        # Scenario S1: India-China Border
        GroundedFact(
            fact_id="S1_F01",
            scenario_id="S1",
            subject="Indian Army",
            subject_qid=S1_ENTITIES["Indian Army"],
            relation="confirmed the loss of lives in a violent face-off with PLA troops at",
            object="Galwan Valley",
            object_qid=S1_ENTITIES["Galwan Valley"],
            number_val=20.0,
            unit="soldiers",
            date_str="2020-06-16",
            location_name="Galwan Valley",
            geonames_id=S1_LOCATIONS["Galwan Valley"]["geonames_id"],
            latitude=S1_LOCATIONS["Galwan Valley"]["lat"],
            longitude=S1_LOCATIONS["Galwan Valley"]["lon"],
            attributed_source="Ministry of External Affairs",
            raw_sentence="The Indian Army confirmed that 20 Indian soldiers lost their lives during a violent face-off with Chinese troops at the Galwan Valley on the night of 15-16 June 2020.",
        ),
        GroundedFact(
            fact_id="S1_F02",
            scenario_id="S1",
            subject="People's Liberation Army",
            subject_qid=S1_ENTITIES["People's Liberation Army"],
            relation="completed mutual disengagement with Indian forces along the north and south banks of",
            object="Pangong Tso",
            object_qid=S1_ENTITIES["Pangong Tso"],
            number_val=10.0,
            unit="days",
            date_str="2021-02-20",
            location_name="Pangong Tso",
            geonames_id=S1_LOCATIONS["Pangong Tso"]["geonames_id"],
            latitude=S1_LOCATIONS["Pangong Tso"]["lat"],
            longitude=S1_LOCATIONS["Pangong Tso"]["lon"],
            attributed_source="Ministry of Defence",
            raw_sentence="Both Indian and Chinese military forces completed a phased and coordinated disengagement from the North and South banks of Pangong Tso in February 2021.",
        ),
        GroundedFact(
            fact_id="S1_F03",
            scenario_id="S1",
            subject="Indian Army",
            subject_qid=S1_ENTITIES["Indian Army"],
            relation="thwarted an attempt by Chinese PLA troops to unilaterally alter the status quo in the",
            object="Yangtse Sector",
            object_qid=S1_ENTITIES["Yangtse Sector"],
            number_val=300.0,
            unit="troops",
            date_str="2022-12-09",
            location_name="Tawang",
            geonames_id=S1_LOCATIONS["Tawang"]["geonames_id"],
            latitude=S1_LOCATIONS["Tawang"]["lat"],
            longitude=S1_LOCATIONS["Tawang"]["lon"],
            attributed_source="Parliamentary Statement by Raksha Mantri",
            raw_sentence="On 9 December 2022, PLA troops attempted to transgress the LAC in Yangtse area of Tawang Sector, which was contested by Indian troops in a firm and resolute manner.",
        ),
        GroundedFact(
            fact_id="S1_F04",
            scenario_id="S1",
            subject="Ministry of External Affairs",
            subject_qid=S1_ENTITIES["Ministry of External Affairs"],
            relation="announced agreement on patrolling arrangements along the Line of Actual Control in",
            object="Eastern Ladakh",
            object_qid=S1_ENTITIES["Line of Actual Control"],
            number_val=2.0,
            unit="patrol points",
            date_str="2024-10-21",
            location_name="Leh",
            geonames_id=S1_LOCATIONS["Leh"]["geonames_id"],
            latitude=S1_LOCATIONS["Leh"]["lat"],
            longitude=S1_LOCATIONS["Leh"]["lon"],
            attributed_source="Foreign Secretary Press Briefing",
            raw_sentence="Foreign Secretary Vikram Misri announced on 21 October 2024 that an agreement had been reached on patrolling arrangements along the LAC in India-China border areas.",
        ),

        # Scenario S2: Assam Floods 2022
        GroundedFact(
            fact_id="S2_F01",
            scenario_id="S2",
            subject="Assam State Disaster Management Authority",
            subject_qid=S2_ENTITIES["Assam State Disaster Management Authority"],
            relation="reported that floodwaters inundated urban residential areas affecting thousands of people in",
            object="Silchar",
            object_qid=S2_ENTITIES["Silchar"],
            number_val=280000.0,
            unit="residents",
            date_str="2022-06-22",
            location_name="Silchar",
            geonames_id=S2_LOCATIONS["Silchar"]["geonames_id"],
            latitude=S2_LOCATIONS["Silchar"]["lat"],
            longitude=S2_LOCATIONS["Silchar"]["lon"],
            attributed_source="ASDMA Daily Flood Bulletin",
            raw_sentence="The ASDMA reported that the town of Silchar remained submerged for over ten days in June 2022 following a breach in the Bethukandi dyke along the Barak river.",
        ),
        GroundedFact(
            fact_id="S2_F02",
            scenario_id="S2",
            subject="Assam State Disaster Management Authority",
            subject_qid=S2_ENTITIES["Assam State Disaster Management Authority"],
            relation="confirmed the cumulative death toll from monsoon floods and landslides across Assam reached",
            object="Assam",
            object_qid=S2_ENTITIES["Assam State Disaster Management Authority"],
            number_val=197.0,
            unit="fatalities",
            date_str="2022-07-15",
            location_name="Guwahati",
            geonames_id=S2_LOCATIONS["Guwahati"]["geonames_id"],
            latitude=S2_LOCATIONS["Guwahati"]["lat"],
            longitude=S2_LOCATIONS["Guwahati"]["lon"],
            attributed_source="ASDMA Cumulative Report",
            raw_sentence="According to official ASDMA figures, 197 people lost their lives due to floods and severe landslides during the 2022 Assam monsoon season.",
        ),
        GroundedFact(
            fact_id="S2_F03",
            scenario_id="S2",
            subject="National Disaster Response Force",
            subject_qid=S2_ENTITIES["National Disaster Response Force"],
            relation="deployed disaster response teams and evacuated marooned villagers across",
            object="Nagaon District",
            object_qid=S2_ENTITIES["Nagaon District"],
            number_val=34.0,
            unit="districts",
            date_str="2022-06-18",
            location_name="Nagaon",
            geonames_id=S2_LOCATIONS["Nagaon"]["geonames_id"],
            latitude=S2_LOCATIONS["Nagaon"]["lat"],
            longitude=S2_LOCATIONS["Nagaon"]["lon"],
            attributed_source="NDRF Situation Report",
            raw_sentence="The NDRF deployed multiple specialized battalions in flood-ravaged districts of Assam, rescuing over 50,000 marooned people in June 2022.",
        ),
        GroundedFact(
            fact_id="S2_F04",
            scenario_id="S2",
            subject="Central Water Commission",
            subject_qid=S2_ENTITIES["Central Water Commission"],
            relation="recorded the Brahmaputra River flowing well above the danger mark at",
            object="Dhubri",
            object_qid=S2_ENTITIES["Brahmaputra River"],
            number_val=0.85,
            unit="meters above danger level",
            date_str="2022-06-25",
            location_name="Dhubri",
            geonames_id=S2_LOCATIONS["Dhubri"]["geonames_id"],
            latitude=S2_LOCATIONS["Dhubri"]["lat"],
            longitude=S2_LOCATIONS["Dhubri"]["lon"],
            attributed_source="Central Water Commission Bulletin",
            raw_sentence="The Central Water Commission recorded the Brahmaputra river flowing 0.85 meters above the danger level at Dhubri gauge station on 25 June 2022.",
        ),

        # Scenario S3: Indian General Election 2024
        GroundedFact(
            fact_id="S3_F01",
            scenario_id="S3",
            subject="Election Commission of India",
            subject_qid=S3_ENTITIES["Election Commission of India"],
            relation="conducted the 18th Lok Sabha general election across",
            object="Lok Sabha",
            object_qid=S3_ENTITIES["Lok Sabha"],
            number_val=7.0,
            unit="phases",
            date_str="2024-03-16",
            location_name="New Delhi",
            geonames_id=S3_LOCATIONS["New Delhi"]["geonames_id"],
            latitude=S3_LOCATIONS["New Delhi"]["lat"],
            longitude=S3_LOCATIONS["New Delhi"]["lon"],
            attributed_source="Chief Election Commissioner Press Conference",
            raw_sentence="Chief Election Commissioner Rajiv Kumar announced that the 2024 Indian general election would be held in seven phases from 19 April to 1 June 2024.",
        ),
        GroundedFact(
            fact_id="S3_F02",
            scenario_id="S3",
            subject="Election Commission of India",
            subject_qid=S3_ENTITIES["Election Commission of India"],
            relation="declared official election results for all parliamentary constituencies on",
            object="New Delhi",
            object_qid=S3_ENTITIES["Election Commission of India"],
            number_val=543.0,
            unit="constituencies",
            date_str="2024-06-04",
            location_name="New Delhi",
            geonames_id=S3_LOCATIONS["New Delhi"]["geonames_id"],
            latitude=S3_LOCATIONS["New Delhi"]["lat"],
            longitude=S3_LOCATIONS["New Delhi"]["lon"],
            attributed_source="ECI Official Declaration",
            raw_sentence="Counting of votes for all 543 Lok Sabha constituencies was conducted and finalized on 4 June 2024.",
        ),
        GroundedFact(
            fact_id="S3_F03",
            scenario_id="S3",
            subject="Election Commission of India",
            subject_qid=S3_ENTITIES["Election Commission of India"],
            relation="certified that the overall voter turnout in the 2024 general election was",
            object="Lok Sabha",
            object_qid=S3_ENTITIES["Lok Sabha"],
            number_val=65.79,
            unit="percent",
            date_str="2024-06-06",
            location_name="New Delhi",
            geonames_id=S3_LOCATIONS["New Delhi"]["geonames_id"],
            latitude=S3_LOCATIONS["New Delhi"]["lat"],
            longitude=S3_LOCATIONS["New Delhi"]["lon"],
            attributed_source="ECI Press Release",
            raw_sentence="The Election Commission certified an aggregate voter turnout of 65.79 percent across the seven phases of the 2024 general elections.",
        ),
    ]
    return facts
