"""
medical_ner.py
Hybrid Medical NER (HuggingFace Clinical NER + rules)
- Uses samrawal/bert-base-uncased_clinical-ner
- Cleans messy PMH strings (glued diseases, hx/sp/tahbso, etc.)
- Adds small dictionary-based entities for important phrases.
"""

from typing import List, Dict
import re

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ---------------------------------------------------------------------
# 1) MODEL SETUP
# ---------------------------------------------------------------------

MODEL_NAME = "samrawal/bert-base-uncased_clinical-ner"

print(f"[INFO] Loading HF clinical NER model: {MODEL_NAME}")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

_ner = pipeline(
    "ner",
    model=_model,
    tokenizer=_tokenizer,
    aggregation_strategy="simple",  # groups sub-tokens into spans
)

# Map HF labels to our simpler schema
LABEL_MAP = {
    "PROBLEM": "SYMPTOM",
    "TREATMENT": "TREATMENT",
    "TEST": "TEST",
}

# ---------------------------------------------------------------------
# 2) CLEANING + GLUED TEXT HANDLING
# ---------------------------------------------------------------------

STOPWORDS = set(
    """
    a an the and or of to for with on in at by from no not none denies deny
    denied without is are was were be been being has have had hx past history current currently
    since as per than then pt patient mild severe moderate acute chronic day days week weeks
    month months year years today yesterday status post
    """.split()
)

GLUED_SPLIT_TERMS = [
    "hypertension",
    "hypothyroidism",
    "hypercholesterolemia",
    "hyponatremia",
    "siadh",
    "diverticulosis",
    "stenosis",
    "cataracts",
    "rhinitis",
    "osteoporosis",
    "glaucoma",
    "eczema",
    "ulcers",
    "scoliosis",
    "hemorrhoids",
    "prolapse",
    "bleeding",
    # extra for your longer notes
    "dyslipidemia",
    "cva",
    "sputum",
    "cytology",
    "groundglass",
    "widely",
    "disseminated",
]

CLINICAL_NOISE = [
    "hx",    # history
    "sp",    # status post / shorthand
    "s/p",
    "tahbso",
]


def clean_pmh_text(raw_text: str) -> str:
    """
    PRE-NER preprocessing:
    - lowercasing
    - split glued disease names using GLUED_SPLIT_TERMS
    - remove some clinical noise tokens (hx, sp, tahbso, etc.)
    - normalize whitespace
    """
    if not isinstance(raw_text, str):
        return ""

    text = raw_text.lower()

    # Remove obvious noise tokens
    for noise in CLINICAL_NOISE:
        text = re.sub(rf"\b{re.escape(noise)}\b", " ", text)

    # Insert spaces around common disease tokens so that
    # 'hypertensionhypothyroidism' -> ' hypertension hypothyroidism '
    for term in GLUED_SPLIT_TERMS:
        text = re.sub(re.escape(term), f" {term} ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# 3) SMALL DICTIONARY BOOST (for key phrases)
# ---------------------------------------------------------------------

# dictionary phrases we really care about in PMH
DICT_PHRASES = {
    "siadh": "SYMPTOM",                 # (or DISEASE if you prefer)
    "memory loss": "SYMPTOM",
    "hemorrhoids": "SYMPTOM",
    "low back pain": "SYMPTOM",
    "postmenopausal bleeding": "SYMPTOM",
    "bladder prolapse": "SYMPTOM",
}


def _find_all_occurrences(text: str, phrase: str):
    pat = re.compile(re.escape(phrase), re.IGNORECASE)
    for m in pat.finditer(text):
        yield m.start(), m.end()


# ---------------------------------------------------------------------
# 4) MAIN NER FUNCTION
# ---------------------------------------------------------------------

def run_medical_ner(raw_text: str) -> List[Dict]:
    """
    Run medical NER on (possibly messy) PMH-like text.
    Returns a list of dicts:
        {
            "entity": str,
            "label": "SYMPTOM" | "TREATMENT" | "TEST",
            "start_offset": int,   # offset in CLEANED text
            "end_offset": int,
        }
    Note: offsets are relative to the cleaned text, not raw.
    For graph building, usually only 'entity' and 'label' matter.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []

    # 1) Clean text first (handles glued diseases, hx, sp, etc.)
    text = clean_pmh_text(raw_text)

    if not text:
        return []

    # 2) HF clinical NER
    outputs = _ner(text)

    spans = []
    for o in outputs:
        raw_group = o.get("entity_group", "").upper()
        label = LABEL_MAP.get(raw_group, "SYMPTOM")
        start = int(o.get("start", 0))
        end = int(o.get("end", start))
        spans.append({"start": start, "end": end, "label": label})

    # sort by start
    spans.sort(key=lambda x: x["start"])

    # merge overlapping / touching spans with same label
    merged = []
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        last = merged[-1]
        if span["label"] == last["label"] and span["start"] <= last["end"]:
            last["end"] = max(last["end"], span["end"])
        else:
            merged.append(span)

    # 3) dictionary-based entities (siadh, memory loss, etc.) on CLEANED text
    dict_spans = []
    for phrase, label in DICT_PHRASES.items():
        for s, e in _find_all_occurrences(text, phrase):
            dict_spans.append({"start": s, "end": e, "label": label})

    # 4) combine HF spans + dict spans, preferring DICT spans on overlap
    # put dict_spans first so they get picked first
    all_spans = dict_spans + merged

    # process from left to right
    all_spans.sort(key=lambda x: x["start"])

    final_spans = []
    occupied = []

    def overlaps(a_start, a_end, b_start, b_end):
        return not (a_end <= b_start or b_end <= a_start)

    for sp in all_spans:
        s, e = sp["start"], sp["end"]
        if any(overlaps(s, e, os, oe) for os, oe in occupied):
            continue
        occupied.append((s, e))
        final_spans.append(sp)

    # 5) build final entities from text
    entities: List[Dict] = []
    for sp in sorted(final_spans, key=lambda x: x["start"]):
        s, e, label = sp["start"], sp["end"], sp["label"]
        phrase = text[s:e]

        # basic cleaning
        phrase_clean = phrase.replace("##", "")
        phrase_clean = re.sub(r"\s+", " ", phrase_clean).strip()

        # strip leading stopwords like "of", "the", "a"
        phrase_clean = re.sub(r"^(of|the|a|an)\s+", "", phrase_clean)

        # strip trailing 'for'
        phrase_clean = re.sub(r"\s+for$", "", phrase_clean)

        if not phrase_clean:
            continue

        # length-based + noise filtering
        if len(phrase_clean) <= 2 and phrase_clean.lower() not in {"ct", "pt"}:
            # drop junk like "6", "ion", single letters
            continue

        if phrase_clean.lower() in STOPWORDS:
            continue

        # optional: drop very generic entities
        NOISE_ENTITIES = {
            "worse",
            "known disease",
        }
        if phrase_clean.lower() in NOISE_ENTITIES:
            continue

        entities.append(
            {
                "entity": phrase_clean,
                "label": label,
                "start_offset": s,
                "end_offset": e,
            }
        )

    # 6) deduplicate by (span, label)
    unique = []
    seen = set()
    for ent in entities:
        key = (ent["start_offset"], ent["end_offset"], ent["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(ent)

    return unique


# ---------------------------------------------------------------------
# 5) Quick CLI test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    t1 = (
        "hypertensionhypothyroidismhypercholesterolemiahyponatremia suspected "
        "siadhdiverticulosislow back pain hx spinal stenosiscataracts allergic "
        "rhinitisosteoporosisglaucomaeczemagastric ulcerscoliosismemory losshemorrhoids  "
        "sp tahbso for postmenopausal bleedingbladder prolapse sp suspension"
    )
    print("RAW 1:")
    print(t1)
    print("\nENTITIES 1:")
    for e in run_medical_ner(t1):
        print(e)

    print("\n" + "="*60 + "\n")

    t2 = (
        "CAD s/p CABG, CVA, CKD, COPD. Stage IV nonsmall cell lung cancer with multiple "
        "intrapulmonary lesions. Chest CT and MRI brain done. Unresolving rightsided "
        "pulmonary infiltrate, increased creatinine levels, cytopenias."
    )
    print("RAW 2:")
    print(t2)
    print("\nENTITIES 2:")
    for e in run_medical_ner(t2):
        print(e)
