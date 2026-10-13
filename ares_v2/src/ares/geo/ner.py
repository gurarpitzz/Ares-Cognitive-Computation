"""Named Entity Recognition (NER) module for ARES Geopolitical & Contextual Intelligence.

Supports spaCy extraction, entity normalization, CoNLL-2003 benchmark evaluation,
and Table IV regeneration (Claim C007).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard mapping from spaCy labels to 4-class CoNLL-style entity categories
SPACY_TO_CONLL = {
    "PERSON": "PER",
    "PER": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "MISC",
    "EVENT": "MISC",
    "LAW": "MISC",
    "WORK_OF_ART": "MISC",
    "PRODUCT": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "MISC",
    "TIME": "MISC",
    "MONEY": "MISC",
    "QUANTITY": "MISC",
    "ORDINAL": "MISC",
    "CARDINAL": "MISC",
}


@dataclass
class ExtractedEntity:
    """Represents an extracted named entity."""
    text: str
    label: str
    conll_label: str
    start_char: int
    end_char: int
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "conll_label": self.conll_label,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": round(self.confidence, 4),
        }


class NamedEntityRecognizer:
    """Wrapper around spaCy NER pipeline for geopolitical entity extraction."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None

    def _load_model(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                logger.warning("spaCy model %s not found. Falling back to blank English.", self.model_name)
                self._nlp = spacy.blank("en")
        return self._nlp

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract all named entities from an input string."""
        if not text or not text.strip():
            return []
        nlp = self._load_model()
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            conll_tag = SPACY_TO_CONLL.get(ent.label_, "MISC")
            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                conll_label=conll_tag,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0,
            ))
        return entities

    def extract_locations(self, text: str) -> List[str]:
        """Extract location and geopolitical entity mentions."""
        ents = self.extract_entities(text)
        return [e.text for e in ents if e.conll_label == "LOC" or e.label in ("GPE", "LOC", "FAC")]

    def extract_organizations(self, text: str) -> List[str]:
        """Extract organization mentions."""
        ents = self.extract_entities(text)
        return [e.text for e in ents if e.conll_label == "ORG"]

    def extract_persons(self, text: str) -> List[str]:
        """Extract person mentions."""
        ents = self.extract_entities(text)
        return [e.text for e in ents if e.conll_label == "PER"]


def evaluate_conll2003(
    ner_model: Optional[NamedEntityRecognizer] = None,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate NER pipeline on the CoNLL-2003 test set using seqeval."""
    from datasets import load_dataset
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    if ner_model is None:
        ner_model = NamedEntityRecognizer("en_core_web_sm")
    nlp = ner_model._load_model()

    ds = load_dataset("lhoestq/conll2003", split=split)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    id2tag = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []

    for item in ds:
        tokens = item["tokens"]
        gold_tag_ids = item["ner_tags"]
        gold_tags = [id2tag.get(tid, "O") for tid in gold_tag_ids]
        y_true.append(gold_tags)

        # Reconstruct sentence and align spaCy tokens
        sentence = " ".join(tokens)
        doc = nlp(sentence)

        # Align predicted spans to gold token boundaries
        pred_tags = ["O"] * len(tokens)
        for ent in doc.ents:
            ent_type = SPACY_TO_CONLL.get(ent.label_, "MISC")
            ent_start = ent.start_char
            ent_end = ent.end_char

            curr_pos = 0
            for i, tok in enumerate(tokens):
                tok_start = curr_pos
                tok_end = curr_pos + len(tok)
                curr_pos = tok_end + 1  # account for space

                # Check overlap
                if tok_start >= ent_start and tok_end <= ent_end:
                    prefix = "B-" if (i == 0 or pred_tags[i - 1] == "O" or not pred_tags[i - 1].endswith(ent_type)) else "I-"
                    pred_tags[i] = f"{prefix}{ent_type}"

        y_pred.append(pred_tags)

    p = float(precision_score(y_true, y_pred))
    r = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "n_samples": len(ds),
        "precision": p,
        "recall": r,
        "f1": f1,
        "detailed_report": report,
    }


def regenerate_table_iv(ner_model: Optional[NamedEntityRecognizer] = None) -> List[Dict[str, str]]:
    """Regenerate Table IV (Sample Named Entity Recognition Outputs) from real model outputs.

    Corresponds to Claim C007 in the claims ledger.
    """
    if ner_model is None:
        ner_model = NamedEntityRecognizer("en_core_web_sm")

    # The exact sentences from Table IV of the monograph
    monograph_sentences = [
        ("Elon Musk is the CEO of Tesla.", ["PERSON", "ORGANIZATION"]),
        ("She lives in New York City.", ["LOCATION"]),
        ("Tesla is leading innovation.", ["ORGANIZATION"]),
    ]

    rows = []
    for sentence, target_types in monograph_sentences:
        ents = ner_model.extract_entities(sentence)
        for ent in ents:
            # Format entity type to match manuscript styling
            type_label = ent.label
            if type_label in ("GPE", "LOC"):
                type_name = "LOCATION"
            elif type_label in ("PERSON", "PER"):
                type_name = "PERSON"
            elif type_label == "ORG":
                type_name = "ORGANIZATION"
            else:
                type_name = type_label

            rows.append({
                "Entity Type": type_name,
                "Extracted Entity": ent.text,
                "Original Sentence": f'"{sentence}"',
            })

    return rows
