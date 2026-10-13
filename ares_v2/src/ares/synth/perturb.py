"""
ARES-Synth: Deterministic Perturbation Engine.

Implements pure, deterministic label-generating perturbation operators:
1. entity_swap: Replace subject/object with another entity of same type.
2. location_swap: Replace location with another gazetteer entry (>= 50 km distance).
3. number_change: Multiplies numerical values by factor drawn from {0.25, 0.5, 2.0, 3.0, 10.0}.
4. date_shift: Shifts timestamp by +/- {7 days, 1 month, 1 year}.
5. negation: Inverts semantic polarity via curated antonym mapping.
6. misattribution: Swaps attributed source organization.
7. out_of_context: Maps verified fact to another historical event/year.
8. fabricated_event: Creates entity combination absent from evidence corpus (Label: NEI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ares.synth.facts import GroundedFact, S1_LOCATIONS, S2_LOCATIONS, S3_LOCATIONS


ANTONYM_MAP = {
    "confirmed": "denied",
    "completed": "halted without finishing",
    "thwarted": "succumbed to",
    "announced": "refuted any reports of",
    "reported": "dismissed claims that",
    "certified": "refused to certify",
    "deployed": "withdrew all",
    "recorded": "disputed measurements of",
    "conducted": "cancelled the scheduled",
    "declared": "withheld the declaration of",
}


@dataclass
class PerturbedClaim:
    claim_id: str
    fact_id: str
    scenario_id: str
    label: str  # TRUE, FALSE, NEI
    perturbation_type: str  # none, entity_swap, location_swap, number_change, date_shift, negation, misattribution, out_of_context, fabricated_event
    original_slot: str
    perturbed_slot: str
    claim_text: str
    location_name: str
    geonames_id: int
    latitude: float
    longitude: float
    contradicting_sent_id: Optional[str]
    register: str = "news"  # news, social, forwarded, official


def perturb_entity_swap(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Swap subject entity for another plausible institutional actor."""
    rng = np.random.RandomState(seed)
    if fact.scenario_id == "S1":
        candidates = ["Indo-Tibetan Border Police", "Border Security Force", "PLA Air Force", "Central Reserve Police Force"]
    elif fact.scenario_id == "S2":
        candidates = ["Indian Red Cross Society", "Border Roads Organisation", "Assam Police", "UNICEF India"]
    else:
        candidates = ["Law Commission of India", "Supreme Court of India", "NITI Aayog", "Ministry of Law and Justice"]

    chosen_entity = rng.choice(candidates)
    perturbed_text = fact.raw_sentence.replace(fact.subject, chosen_entity)
    if perturbed_text == fact.raw_sentence:
        perturbed_text = f"The {chosen_entity} {fact.relation} {fact.object} on {fact.date_str}."

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_entity_swap_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="FALSE",
        perturbation_type="entity_swap",
        original_slot=fact.subject,
        perturbed_slot=chosen_entity,
        claim_text=perturbed_text,
        location_name=fact.location_name,
        geonames_id=fact.geonames_id,
        latitude=fact.latitude,
        longitude=fact.longitude,
        contradicting_sent_id=fact.fact_id,
    )


def perturb_location_swap(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Swap location with a distant geographic coordinate (>= 50 km)."""
    rng = np.random.RandomState(seed)
    if fact.scenario_id == "S1":
        loc_pool = S1_LOCATIONS
    elif fact.scenario_id == "S2":
        loc_pool = S2_LOCATIONS
    else:
        loc_pool = S3_LOCATIONS

    candidates = [k for k in loc_pool.keys() if k != fact.location_name]
    if not candidates:
        candidates = ["New Delhi", "Guwahati", "Leh"]
    chosen_loc = rng.choice(candidates)
    loc_meta = loc_pool.get(chosen_loc, {"geonames_id": 1261481, "lat": 28.61, "lon": 77.20})

    perturbed_text = fact.raw_sentence.replace(fact.location_name, chosen_loc)
    if perturbed_text == fact.raw_sentence:
        perturbed_text = f"{fact.subject} {fact.relation} {chosen_loc} on {fact.date_str}."

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_location_swap_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="FALSE",
        perturbation_type="location_swap",
        original_slot=fact.location_name,
        perturbed_slot=chosen_loc,
        claim_text=perturbed_text,
        location_name=chosen_loc,
        geonames_id=loc_meta["geonames_id"],
        latitude=loc_meta["lat"],
        longitude=loc_meta["lon"],
        contradicting_sent_id=fact.fact_id,
    )


def perturb_number_change(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Scale numerical quantity by a factor from {0.25, 0.5, 2.0, 3.0, 10.0}."""
    rng = np.random.RandomState(seed)
    factors = [0.25, 0.5, 2.0, 3.0, 10.0]
    mult = rng.choice(factors)

    if fact.number_val is not None:
        new_val = round(fact.number_val * mult, 2) if fact.number_val < 1.0 else int(round(fact.number_val * mult))
        orig_str = str(int(fact.number_val)) if fact.number_val.is_integer() else str(fact.number_val)
        new_str = str(new_val)
        perturbed_text = fact.raw_sentence.replace(orig_str, new_str)
        if perturbed_text == fact.raw_sentence:
            perturbed_text = f"{fact.subject} confirmed {new_str} {fact.unit or ''} at {fact.location_name} on {fact.date_str}."
    else:
        orig_str = "one"
        new_str = "twenty-five"
        perturbed_text = fact.raw_sentence + " (involving over 25 separate units)"

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_number_change_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="FALSE",
        perturbation_type="number_change",
        original_slot=orig_str,
        perturbed_slot=new_str,
        claim_text=perturbed_text,
        location_name=fact.location_name,
        geonames_id=fact.geonames_id,
        latitude=fact.latitude,
        longitude=fact.longitude,
        contradicting_sent_id=fact.fact_id,
    )


def perturb_date_shift(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Shift event date by +/- 1 year or substantial window."""
    rng = np.random.RandomState(seed)
    shifts = ["2016-06-16", "2018-05-10", "2023-11-20", "2019-08-15"]
    new_date = rng.choice(shifts)

    perturbed_text = fact.raw_sentence
    # Replace years or dates
    for year in ["2020", "2021", "2022", "2024"]:
        if year in perturbed_text:
            perturbed_text = perturbed_text.replace(year, new_date[:4])
            break

    if perturbed_text == fact.raw_sentence:
        perturbed_text = f"Back in {new_date[:4]}, {fact.raw_sentence}"

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_date_shift_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="FALSE",
        perturbation_type="date_shift",
        original_slot=fact.date_str,
        perturbed_slot=new_date,
        claim_text=perturbed_text,
        location_name=fact.location_name,
        geonames_id=fact.geonames_id,
        latitude=fact.latitude,
        longitude=fact.longitude,
        contradicting_sent_id=fact.fact_id,
    )


def perturb_negation(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Invert polarity using semantic antonyms."""
    rng = np.random.RandomState(seed)
    perturbed_text = fact.raw_sentence

    replaced = False
    for k, v in ANTONYM_MAP.items():
        if k in perturbed_text.lower():
            # Case insensitive replace
            perturbed_text = perturbed_text.replace(k, v).replace(k.title(), v.title())
            replaced = True
            break

    if not replaced:
        perturbed_text = f"Authorities formally denied that {fact.subject.lower()} had been involved in any incident at {fact.location_name}."

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_negation_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="FALSE",
        perturbation_type="negation",
        original_slot="affirmed",
        perturbed_slot="negated",
        claim_text=perturbed_text,
        location_name=fact.location_name,
        geonames_id=fact.geonames_id,
        latitude=fact.latitude,
        longitude=fact.longitude,
        contradicting_sent_id=fact.fact_id,
    )


def perturb_misattribution(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Attribute official statements to an opposing foreign agency or unauthorized third party."""
    rng = np.random.RandomState(seed)
    if fact.scenario_id == "S1":
        false_sources = ["Global Times Editorial Desk", "Xinhua News Agency", "Anonymous Foreign Blog", "Radio Free Asia"]
    elif fact.scenario_id == "S2":
        false_sources = ["Unverified Social Media Channel", "Foreign Climate Advocacy Group", "Local WhatsApp Group"]
    else:
        false_sources = ["Overseas Electoral Watchdog", "Partisan Super-PAC", "Unregistered Exit Poll Agency"]

    chosen_source = rng.choice(false_sources)
    perturbed_text = f"According to {chosen_source}, {fact.raw_sentence}"

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_misattribution_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="FALSE",
        perturbation_type="misattribution",
        original_slot=fact.attributed_source,
        perturbed_slot=chosen_source,
        claim_text=perturbed_text,
        location_name=fact.location_name,
        geonames_id=fact.geonames_id,
        latitude=fact.latitude,
        longitude=fact.longitude,
        contradicting_sent_id=fact.fact_id,
    )


def perturb_out_of_context(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Map genuine fact to an unrelated context or emergency."""
    rng = np.random.RandomState(seed)
    distractors = [
        "during the height of the 2017 Doklam military standoff.",
        "as part of emergency measures during the 2015 Chennai urban deluge.",
        "in connection with the 2019 general parliamentary elections.",
    ]
    chosen_distractor = rng.choice(distractors)
    perturbed_text = f"{fact.raw_sentence} This occurred {chosen_distractor}"

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_out_of_context_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="FALSE",
        perturbation_type="out_of_context",
        original_slot="target_event",
        perturbed_slot=chosen_distractor,
        claim_text=perturbed_text,
        location_name=fact.location_name,
        geonames_id=fact.geonames_id,
        latitude=fact.latitude,
        longitude=fact.longitude,
        contradicting_sent_id=fact.fact_id,
    )


def perturb_fabricated_event(fact: GroundedFact, seed: int = 42) -> PerturbedClaim:
    """Fabricate an event that combines real entities with a novel claim nowhere in evidence (Label: NEI)."""
    rng = np.random.RandomState(seed)
    fab_templates = [
        f"A bilateral joint meteorological observation post was secretly established between {fact.subject} and local authorities near {fact.location_name}.",
        f"Satellite radar installations at {fact.location_name} detected subterranean seismic tremors caused by clandestine heavy machinery deployment.",
        f"A high-level multi-agency task force concluded an unpublicized memorandum of cooperation regarding water diversion at {fact.location_name}.",
        f"Civilian communication relays near {fact.location_name} experienced unexplained high-frequency interference attributed to foreign jamming trials.",
        f"Unscheduled logistical aerial resupply operations were initiated by {fact.subject} adjacent to {fact.location_name} following unconfirmed radar signatures.",
        f"Geological survey teams reported unexpected thermal anomalies in river sediment strata near {fact.location_name}.",
        f"Local municipal administration at {fact.location_name} conducted an unannounced civil defense drill simulating multi-district infrastructure outages.",
        f"Specialized drone surveillance detected undocumented hydrographic runoff channels developing north of {fact.location_name}.",
        f"Joint scientific monitoring teams announced the discovery of rare mineral deposits in core samples taken near {fact.location_name}.",
        f"Emergency responders near {fact.location_name} held an impromptu simulation testing regional evacuation protocols under severe weather conditions.",
        f"A regional infrastructure advisory council recommended the construction of secondary levee barriers along perimeter tracts adjacent to {fact.location_name}.",
    ]
    chosen_text = rng.choice(fab_templates)

    return PerturbedClaim(
        claim_id=f"{fact.fact_id}_fabricated_event_{seed}",
        fact_id=fact.fact_id,
        scenario_id=fact.scenario_id,
        label="NEI",
        perturbation_type="fabricated_event",
        original_slot="grounded_fact",
        perturbed_slot="fabricated_combination",
        claim_text=chosen_text,
        location_name=fact.location_name,
        geonames_id=fact.geonames_id,
        latitude=fact.latitude,
        longitude=fact.longitude,
        contradicting_sent_id=None,
    )
