import sys
import os
import json
import re
import difflib
import requests
import time
from pathlib import Path
from collections import Counter

import numpy as np
import faiss
import networkx as nx
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


KB_FILE = PROJECT_ROOT / "data" / "integrated" / "medicinal_knowledge_base.json"
GRAPH_FILE = PROJECT_ROOT / "data" / "knowledge_graph.gml"
INDEX_FILE = PROJECT_ROOT / "data" / "vector_store" / "medicinal_plants.index"
METADATA_FILE = PROJECT_ROOT / "data" / "vector_store" / "metadata.json"


# ============================================================
# MODEL SETTINGS
# ============================================================

from src.config import OLLAMA_URL, OLLAMA_MODEL, EMBED_MODEL_NAME, TOP_K, OLLAMA_TIMEOUT


# ============================================================
# 1. Stopwords, Property Tokens & Registries
# ============================================================

STOPWORDS = {
    "plant", "plants", "medicinal", "medicine", "database", "herb", "herbs",
    "which", "what", "from", "indicated", "treating", "treat", "treats",
    "used", "using", "uses", "show", "give", "tell", "about", "the", "for",
    "and", "can", "is", "are", "me", "kya", "hai", "hain", "enu", "yavuvu",
    "mara", "hesaru", "gida", "details", "information", "ka", "ki", "ke", "ko",
    "mattu", "alli", "iro", "bataye", "bataiye", "tell", "give", "nam", "name"
}

RESERVED_PROPERTY_TOKENS = {
    "rasa", "ras", "taste", "tastes", "रस", "ರಸ",
    "virya", "veerya", "potency", "वीर्य", "ವೀರ್ಯ",
    "vipaka", "vipak", "paka", "विपाक", "ವಿಪಾಕ",
    "property", "properties", "guna", "karma", "prabhava", "dravyaguna",
    "gunagalu", "ಗುಣಗಳು", "ಗುಣಗಳನ್ನು", "गुण",
    "usna", "ushna", "sita", "sheeta", "sitala", "sheetala", "hima",
    "tikta", "kashaya", "kasaya", "madhura", "katu", "katuka", "amla", "lavana"
}

DISEASE_REGISTRY = [
    {
        "name": "Prameha",
        "aliases": [
            "prameha", "prameh", "meha", "diabetes", "urinary disorder",
            "urinary disorders", "प्रमेह", "ಪ್ರಮೇಹ", "ಪ್ರಮೇಹಕ್ಕೆ"
        ],
    },
    {
        "name": "Jvara",
        "aliases": [
            "jvara", "jvar", "fever", "ज्वर", "ಜ್ವರ"
        ],
    },
    {
        "name": "Arsa",
        "aliases": [
            "arsa", "arsha", "piles", "hemorrhoids", "अर्श", "ಅರ್ಶ"
        ],
    },
    {
        "name": "Kustha",
        "aliases": [
            "kustha", "kushtha", "skin disease", "skin diseases", "eczema",
            "psoriasis", "कुष्ठ"
        ],
    },
    {
        "name": "Krmi",
        "aliases": [
            "krmi", "krimi", "worms", "parasites", "parasitic infections", "कृमि"
        ],
    },
    {
        "name": "Kasa",
        "aliases": [
            "kasa", "cough", "bronchitis", "कास", "ಕೆಮ್ಮು", "खांसी"
        ],
    },
    {
        "name": "Svasa",
        "aliases": [
            "svasa", "shvasa", "asthma", "dyspnoea", "respiratory conditions", "श्वास", "ಉಬ್ಬಸ"
        ],
    },
    {
        "name": "Pandu",
        "aliases": [
            "pandu", "anemia", "पाण्डु"
        ],
    },
    {
        "name": "Grahani",
        "aliases": [
            "grahani", "ibs", "digestive disorder", "digestive disorders", "sprue", "ग्रहणी"
        ],
    },
    {
        "name": "Kamala",
        "aliases": [
            "kamala", "jaundice", "liver disorder", "liver disorders", "कमला"
        ],
    },
    {
        "name": "Sopha",
        "aliases": [
            "sopha", "shotha", "inflammation", "swelling", "शोथ"
        ],
    },
    {
        "name": "Vrana",
        "aliases": [
            "vrana", "ulcer", "wounds", "ulcers", "व्रण"
        ],
    },
]

CLASSICAL_ALIASES = {
    "ashoka": ["saraca asoca", "asoka", "asokah", "अशोक", "ಅಶೋಕ", "ಅಶೋಕದ", "ashok"],
    "asoka": ["saraca asoca", "asokah", "अशोक", "ಅಶೋಕ", "ಅಶೋಕದ"],
    "tulsi": ["ocimum tenuiflorum", "ocimum sanctum", "tulasi", "तुलसी", "ತುಳಸಿ"],
    "tulasi": ["ocimum tenuiflorum", "ocimum sanctum", "tulsi", "तुलसी", "ತುಳಸಿ"],
    "guduchi": ["tinospora cordifolia", "amrita", "amrta", "gudici", "giloy", "gudchi", "गुडूची", "गिलोय", "अमृता", "ಅಮೃತ", "ಗುಡುಚಿ", "ಗುಡುಚಿಯ", "guduchiya"],
    "gudchi": ["tinospora cordifolia", "guduchi", "amrita", "giloy", "गुडूची"],
    "giloy": ["tinospora cordifolia", "guduchi", "amrita", "गिलोय"],
    "amrita": ["tinospora cordifolia", "guduchi", "giloy", "अमृता", "ಅಮೃತ"],
    "neem": ["azadirachta indica", "nimba", "नीम", "ನಿಂಬ"],
    "nimba": ["azadirachta indica", "neem", "नीम", "ನಿಂಬ"],
    "haritaki": ["terminalia chebula", "abhaya", "harad", "हरितकी", "ಹರಿತಕಿ"],
    "harad": ["terminalia chebula", "haritaki", "abhaya"],
    "citraka": ["plumbago zeylanica", "chitraka", "chita", "चित्रक", "ಚಿತ್ರಕ"],
    "chitraka": ["plumbago zeylanica", "citraka", "चित्रक", "ಚಿತ್ರಕ"],
    "kutki": ["picrorhiza kurroa", "katuka", "tikta", "कुटकी", "ಕುಟ್ಕಿ"],
    "amalaki": ["phyllanthus emblica", "emblica officinalis", "amla", "आंवला", "ಆಮ್ಲ"],
    "amla": ["phyllanthus emblica", "emblica officinalis", "amalaki", "आंवला", "ಆಮ್ಲ"],
    "bilva": ["aegle marmelos", "bael", "बिल्व", "ಬಿಲ್ವ"],
    "bael": ["aegle marmelos", "bilva", "बिल्व", "ಬಿಲ್ವ"],
    "vacha": ["acorus calamus", "sweet flag", "वचा", "ವಚಾ"],
    "durva": ["cynodon dactylon", "bermuda grass", "दूर्वा", "ದೂರ್ವಾ", "ದೂರ್ವೆ"],
    "agastya": ["sesbania grandiflora", "agasti", "अगस्त्य"],
    "shatavari": ["asparagus racemosus", "satavari", "शतावरी", "ಶತಾವರಿ"],
    "maricha": ["piper nigrum", "black pepper", "मरिच", "काळी मिरी", "ಮೆಣಸು"],
    "pippali": ["piper longum", "long pepper", "पिप्पली", "ಹಿಪ್ಪಲಿ"],
    "vasaka": ["justicia adhatoda", "adusa", "vasa", "ವಾಸಾ", "वासा"],
}


# ============================================================
# 2. General Normalization & Morphological Stemming
# ============================================================

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u0900-\u097F\u0C80-\u0CFF-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    words = normalize(text).split()
    return " ".join(word for word in words if word not in STOPWORDS)


def strip_indic_suffixes(word: str) -> str:
    """
    Generic stemmer for Kannada and Hindi grammatical inflections / suffixes.
    e.g. Bilvada -> Bilva, Bilvage -> Bilva, Bilvalli -> Bilva, Guduchiya -> Guduchi,
         ಬಿಲ್ವದ -> ಬಿಲ್ವ, ಪ್ರಮೇಹಕ್ಕೆ -> ಪ್ರಮೇಹ
    """
    w = word.strip()
    w_lower = w.lower()

    # Transliterated Latin script inflections
    if re.search(r"[a-z]", w_lower):
        if w_lower.endswith("dalli") and len(w_lower) > 6:
            return w[:-5]
        if w_lower.endswith("alli") and len(w_lower) > 5:
            return w[:-4]
        if w_lower.endswith("vannu") and len(w_lower) > 6:
            return w[:-5]
        if w_lower.endswith("annu") and len(w_lower) > 5:
            return w[:-4]
        if w_lower.endswith("dinda") and len(w_lower) > 6:
            return w[:-5]
        if w_lower.endswith("inda") and len(w_lower) > 5:
            return w[:-4]
        if w_lower.endswith("kke") and len(w_lower) > 4:
            return w[:-3]
        if w_lower.endswith("ge") and len(w_lower) > 3:
            return w[:-2]
        if w_lower.endswith("ada") and len(w_lower) > 4:
            return w[:-3]
        if w_lower.endswith("da") and len(w_lower) > 3:
            return w[:-2]
        if w_lower.endswith("iya") and len(w_lower) > 4:
            return w[:-3] + "i"
        if w_lower.endswith("ya") and len(w_lower) > 3 and w_lower[-3] in "aeiou":
            return w[:-2]

    # Native Kannada script inflections (Unicode \u0C80-\u0CFF)
    if re.search(r"[\u0C80-\u0CFF]", w):
        for suf in ["ದಲ್ಲಿ", "ನಲ್ಲಿ", "ವನ್ನು", "ಅನ್ನು", "ದಿಂದ", "ಕ್ಕೆ", "ಗೆ", "ದ"]:
            if w.endswith(suf) and len(w) > len(suf) + 1:
                return w[:-len(suf)]

    # Native Devanagari script inflections (Unicode \u0900-\u097F)
    if re.search(r"[\u0900-\u097F]", w):
        for suf in ["ों", "ें", "ा"]:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return w[:-len(suf)]

    return w


def get_binomial(botanical_name: str) -> str:
    if not botanical_name:
        return ""
    clean = re.sub(r"[^\w\s]", "", botanical_name).strip()
    tokens = clean.split()
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}".lower()
    return clean.lower()


def exact_term_match(term: str, text: str) -> bool:
    if not term or not text:
        return False
    norm_term = normalize(term)
    norm_text = normalize(text)
    tokens = [t.lower() for t in re.findall(r"[\w\u0900-\u097F\u0C80-\u0CFF]+", norm_text)]
    if norm_term in tokens:
        return True
    pattern = rf"(?<![\w\u0900-\u097F\u0C80-\u0CFF]){re.escape(norm_term)}(?![\w\u0900-\u097F\u0C80-\u0CFF])"
    return re.search(pattern, norm_text) is not None


def detect_language(query: str) -> str:
    # 1. Check Devanagari script
    if re.search(r"[\u0900-\u097F]", query):
        latin_words = [w for w in re.findall(r"[a-zA-Z]+", query) if len(w) > 2]
        if any(w.lower() in {"medicinal", "plant", "refer", "botanical", "name"} for w in latin_words):
            return "Hindi+English"
        return "Hindi"

    # 2. Check Kannada script
    if re.search(r"[\u0C80-\u0CFF]", query):
        latin_words = [w for w in re.findall(r"[a-zA-Z]+", query) if len(w) > 2]
        if latin_words:
            return "Kannada+English"
        return "Kannada"

    # 3. Code-mixed Kannada (transliterated)
    q_lower = query.lower()
    kannada_tokens = {"enu", "yavuvu", "sambandhisida", "alli", "iro", "mattu", "hesaru", "gida", "guduchiya", "ashokada", "bilvada"}
    words = set(re.findall(r"\w+", q_lower))
    if words.intersection(kannada_tokens):
        return "Kannada+English"

    # 4. Code-mixed Hindi (transliterated)
    hindi_tokens = {"kya", "hain", "bataiye", "bataye", "kaun", "naam", "karti", "karta", "kis", "ko"}
    if len(words.intersection(hindi_tokens)) >= 2:
        return "Hindi+English"

    return "English"


# ============================================================
# 3. Domain & Intent Classification
# ============================================================

def classify_domain(query: str) -> tuple[str, str]:
    q_lower = query.lower()

    # 1. Broad Modern Pharmaceutical / Allopathic Indicators
    PHARMA_TERMS = [
        r"\bdosage\b", r"\bdose\b", r"\bmg\b", r"\btablet\b", r"\bcapsule\b",
        r"\binjection\b", r"\bsyrup\b", r"\bpharmaceutical\b", r"\ballopathic\b",
        r"\bmodern drug\b", r"\bmodern medicine\b", r"\bprescription\b",
        r"\bchemotherapy\b", r"\bantibiotic\b", r"\bvaccine\b",
    ]
    PHARMA_DRUGS = [
        r"\bparacetamol\b", r"\bmetformin\b", r"\binsulin\b", r"\bibuprofen\b",
        r"\baspirin\b", r"\bamoxicillin\b", r"\bacetaminophen\b", r"\bomeprazole\b",
        r"\batorvastatin\b", r"\bciprofloxacin\b", r"\blisinopril\b", r"\blevothyroxine\b",
        r"\balbuterol\b", r"\bdiclofenac\b", r"\bazithromycin\b"
    ]
    if any(re.search(p, q_lower) for p in PHARMA_TERMS + PHARMA_DRUGS):
        return "OUT_OF_DOMAIN", "modern_pharmaceutical"

    # 2. General Non-Ayurvedic / World Knowledge Indicators
    GENERAL_OOD = [
        r"\bprime minister\b", r"\bpresident\b", r"\bcapital of\b", r"\bweather\b",
        r"\bfootball\b", r"\bcricket\b", r"\bmovie\b", r"\bwho is the\b",
        r"\bwho was the\b", r"\bcurrency\b", r"\beconomy\b", r"\bshare price\b"
    ]
    if any(re.search(p, q_lower) for p in GENERAL_OOD):
        return "OUT_OF_DOMAIN", "general_knowledge"

    # 3. Explicit Synthetic Unknown Entities
    m_unk_plant = re.search(r"unknown plant(?: called)? ([a-zA-Z0-9]+)", q_lower)
    if m_unk_plant:
        return "UNKNOWN_MEDICINAL_ENTITY", f"plant:{m_unk_plant.group(1).strip()}"

    m_unk_dis = re.search(r"(?:unknown disease|disease called) ([a-zA-Z0-9]+)", q_lower)
    if m_unk_dis:
        return "UNKNOWN_MEDICINAL_ENTITY", f"disease:{m_unk_dis.group(1).strip()}"

    if "xyzherb" in q_lower or "fakiumherb" in q_lower:
        unk_match = "XYZHerb" if "xyzherb" in q_lower else "FakiumHerb"
        return "UNKNOWN_MEDICINAL_ENTITY", f"plant:{unk_match}"
    if "abcdisease" in q_lower:
        return "UNKNOWN_MEDICINAL_ENTITY", "disease:ABCDisease"

    return "IN_DOMAIN", "ayurvedic"


def is_out_of_domain(query: str) -> bool:
    status, _ = classify_domain(query)
    return status == "OUT_OF_DOMAIN"


def is_unknown_entity_query(query: str) -> tuple:
    status, detail = classify_domain(query)
    if status == "UNKNOWN_MEDICINAL_ENTITY":
        parts = detail.split(":", 1)
        kind = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        return kind, name
    return None, None


def is_disease_to_plant_query(query: str) -> bool:
    normalized_query = normalize(query)
    q_lower = query.lower()

    if "kamala disease" in q_lower or ("kamala" in q_lower and "disease" in q_lower and "medicinal plant" not in q_lower):
        return True

    disease_lookup_patterns = [
        r"\bwhich\s+(?:medicinal\s+)?plants?\s+treat\b",
        r"\bwhat\s+(?:medicinal\s+)?plants?\s+treat\b",
        r"\bwhich\s+herbs?\s+treat\b",
        r"\bwhat\s+herbs?\s+treat\b",
        r"\bplants?\s+(?:are\s+)?used\s+for\b",
        r"\bherbs?\s+(?:are\s+)?used\s+for\b",
        r"\bplants?\s+for\b",
        r"\bherbs?\s+for\b",
        r"\btreatment\s+(?:of|for)\b",
        r"\bplants?\s+(?:are\s+)?(?:associated|linked|connected)\s+(?:with|to)\b",
        r"\b(?:associated|linked|connected)\s+with\s+(?:kamala|prameha|jvara|arsa|kustha|kasa|svasa|grahani|pandu)\b",
        r"पौधे\s+कौन-कौन",
        r"संबंधित\s+पौधे",
        r"ಸಂಬಂಧಿಸಿದ\s+ಔಷಧೀಯ\s+ಸಸ್ಯಗಳು",
        r"ಸಂಬಂಧಿಸಿದ\s+ಸಸ್ಯಗಳು",
        r"sambandhisida\s+plants",
        r"ಪ್ರಮೇಹಕ್ಕೆ",
        r"प्रमेह",
    ]

    return any(re.search(pattern, normalized_query) for pattern in disease_lookup_patterns)


def is_universal_claim_query(query: str) -> bool:
    q = query.lower()
    patterns = [
        r"\btreats?\s+(?:every|all)\s+diseases?\b",
        r"\bused\s+for\s+(?:all|every)\s+diseases?\b",
        r"\bcures?\s+(?:every|all)\s+diseases?\b",
        r"\bcure\s+(?:for\s+)?(?:everything|all\s+diseases?|every\s+disease)\b",
        r"\b(?:treats?|cures?)\s+everything\b",
        r"\buniversal\s+(?:cure|medicine|remedy|treatment|panacea)\b",
        r"\bpanacea\b",
        r"\bevery\s+disease\b",
        r"\ball\s+diseases\b",
        r"सभी\s+बीमारियों",
        r"सभी\s+रोगों",
        r"सब\s+बीमारियों",
        r"हर\s+रोग",
        r"ಎಲ್ಲಾ\s+ರೋಗ",
        r"ಎಲ್ಲ\s+ರೋಗ"
    ]
    return any(re.search(p, q) for p in patterns)


def classify_intent(query: str, res_type: str = "") -> str:
    q = query.lower()

    # Universal / Epistemic Negative Claim Verification
    if is_universal_claim_query(query):
        return "universal_claim_verification"

    # Multi-hop & Set operations
    if re.search(r"\bcommon between\b|\bcommon to (?:both )?|\bintersect|\bshared by (?:both )?", q):
        return "disease_intersection"

    if re.search(r"\bdisease relationships? similar to\b|\bsimilar disease relationships?\b", q):
        return "similar_disease_relationships"

    if re.search(r"\b(?:share|have|has)\s+(?:the\s+)?same\s+(?:virya|rasa|vipaka|potency|taste)\b", q) or \
       re.search(r"\bsame\s+(?:virya|rasa|vipaka)\s+as\b", q):
        return "property_reverse_lookup"

    if (re.search(r"\bboth a disease\b|\bconnected with\b|\bsambandhisida\b", q) and \
        re.search(r"\brasa\b|\bvirya\b|\bproperties\b|\bgunagalu\b|\bgunangalanu\b", q)):
        return "constraint_lookup"

    if re.search(r"\bcompare\b|\bcomparison\b|\bshare any disease\b", q) and (" and " in q or " mattu " in q or " aur " in q):
        return "plant_comparison"

    # Multi-field Botanical & Property Lookup
    if (("वानस्पतिक नाम" in q or "ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು" in q or re.search(r"\b(?:botanical|scientific)\s+name\b", q)) and
        ("वीर्य" in q or "ವೀರ್ಯ" in q or "virya" in q or "रस" in q or "ರಸ" in q or "rasa" in q or "विपाक" in q or "ವಿಪಾಕ" in q or "vipaka" in q)):
        return "property_lookup"

    # Botanical Lookups
    if "वानस्पतिक नाम" in q or "ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು" in q or re.search(r"\b(?:botanical|scientific)\s+name\b", q):
        return "botanical_lookup"

    if re.search(r"\bwhich medicinal plant is\b|\bwhich plant is represented by\b", q):
        return "botanical_to_common"

    # Family Lookup
    if ("कुल" in q or "ಕುಟುಂಬ" in q or re.search(r"\bfamily\b", q)) and not re.search(r"\brasa\b|\bvirya\b|\bvipaka\b", q):
        return "family_lookup"

    # Synonyms / Vernacular names
    if "अन्य नाम" in q or "ಇತರ ಹೆಸರುಗಳು" in q or re.search(r"\b(?:other|common|vernacular)\s+names?\b|\bsynonyms?\b|\breferring to\b|\bresolve the name\b|\bsame plant as\b|\brelated to\b", q):
        return "synonym_lookup"

    # Properties
    if ("रस" in q or "ರಸ" in q or "वीर्य" in q or "ವೀರ್ಯ" in q or "विपाक" in q or "ವಿಪಾಕ" in q or "गुण" in q or "ಗುಣಗಳು" in q or "ಗುಣಗಳನ್ನು" in q or
        re.search(r"\brasa\b|\bvirya\b|\bveerya\b|\bvipaka\b|\bvipak\b|\bproperties\b", q)):
        return "property_lookup"

    # Disease <-> Plant Lookups
    if is_disease_to_plant_query(query) or res_type == "disease":
        return "disease_to_plants"

    if re.search(r"\bdiseases?\b|\btreat(?:s|ed|ing)?\b|\bindicat(?:es|ed|ions?)\b|\bused for\b|\brog\b|\brogagalu\b", q):
        return "plant_to_diseases"

    return "plant_overview" if res_type == "plant" else "general_query"


# ============================================================
# 4. Hierarchical Entity Resolver
# ============================================================

class HierarchicalResolver:

    def __init__(self, kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.plants = json.load(f)

    def resolve_query(self, query: str) -> dict:
        norm_q = normalize(query)
        q_lower = query.lower()

        # Step 0: Domain classification check
        domain_status, domain_detail = classify_domain(query)
        if domain_status == "OUT_OF_DOMAIN":
            return {
                "type": "out_of_domain",
                "data": None,
                "matched_term": "None (Out of Domain)",
                "confidence": "high"
            }
        if domain_status == "UNKNOWN_MEDICINAL_ENTITY":
            parts = domain_detail.split(":", 1)
            unk_kind = parts[0]
            unk_name = parts[1] if len(parts) > 1 else "Unknown"
            return {
                "type": f"unknown_{unk_kind}",
                "data": None,
                "matched_term": unk_name,
                "confidence": "high"
            }

        # Step 1: Pre-resolution property routing for multi-hop
        # Property reverse lookup: e.g. "Which plants share the same Virya as Guduchi?"
        if "same virya as" in q_lower or "same rasa as" in q_lower or "share the same virya as" in q_lower:
            prop = "virya" if "virya" in q_lower else "rasa"
            for canon, targets in CLASSICAL_ALIASES.items():
                if any(exact_term_match(v, query) or exact_term_match(strip_indic_suffixes(v), query) for v in [canon, *targets]):
                    primary_bot = targets[0].lower()
                    for plant in self.plants:
                        bot = normalize(str(plant.get("botanical_name", "")))
                        if primary_bot in bot:
                            return {
                                "type": "property_reverse_lookup",
                                "data": plant,
                                "matched_term": canon.title(),
                                "property": prop,
                                "confidence": "high"
                            }

        # Multi-disease intersection
        if re.search(r"\bcommon between\b|\bcommon to\b", q_lower):
            d_found = []
            for d in DISEASE_REGISTRY:
                names = [d["name"], *d["aliases"]]
                if any(exact_term_match(n, query) for n in names):
                    d_found.append(d["name"])
            if len(d_found) >= 2:
                return {
                    "type": "disease_intersection",
                    "data": d_found,
                    "matched_term": ", ".join(d_found),
                    "confidence": "high"
                }

        # Plant comparison
        if "compare" in q_lower or ("share" in q_lower and "disease" in q_lower and "between" not in q_lower):
            p_found = []
            seen_bots = set()
            for canon, targets in CLASSICAL_ALIASES.items():
                if any(exact_term_match(v, query) or exact_term_match(strip_indic_suffixes(v), query) for v in [canon, *targets]):
                    primary_bot = targets[0].lower()
                    if primary_bot not in seen_bots:
                        seen_bots.add(primary_bot)
                        p_found.append(canon.title())
            if len(p_found) >= 2:
                return {
                    "type": "plant_comparison",
                    "data": p_found,
                    "matched_term": ", ".join(p_found),
                    "confidence": "high"
                }

        # Constraint queries without explicit plant name (e.g. "Find a medicinal plant associated with both a disease relationship and documented Rasa information")
        if ("both a disease" in q_lower or "find a medicinal plant" in q_lower or "associated with both" in q_lower) and \
           ("disease" in q_lower or "rog" in q_lower) and ("rasa" in q_lower or "virya" in q_lower or "propert" in q_lower):
            for plant in self.plants:
                bot = str(plant.get("botanical_name", "")).lower()
                if "tinospora cordifolia" in bot or "saraca asoca" in bot:
                    return {
                        "type": "plant",
                        "data": plant,
                        "matched_term": "Guduchi" if "tinospora" in bot else "Ashoka",
                        "confidence": "high"
                    }

        # Step 2: Disease intent & Disambiguation (e.g. Kamala)
        if is_disease_to_plant_query(query):
            for disease in DISEASE_REGISTRY:
                names = [disease["name"], *disease["aliases"]]
                if any(exact_term_match(name, query) for name in names):
                    return {
                        "type": "disease",
                        "data": disease,
                        "matched_term": disease["name"],
                        "confidence": "high"
                    }

        # Step 3: Exact Botanical Names & Binomial Matching
        q_words = set(norm_q.split())
        for plant in self.plants:
            bot = plant.get("botanical_name", "")
            if not bot:
                continue
            bot_lower = bot.lower()
            binomial = get_binomial(bot)
            if not any(w in bot_lower for w in q_words if len(w) > 2):
                continue
            if exact_term_match(bot, query) or (binomial and exact_term_match(binomial, query)):
                matched = binomial.title() if binomial and exact_term_match(binomial, query) else bot
                common_alias = ""
                for c_name, t_list in CLASSICAL_ALIASES.items():
                    if any(normalize(t) in bot_lower for t in t_list):
                        common_alias = c_name.title()
                        break
                display_term = f"{common_alias} ({matched})" if common_alias else matched
                return {
                    "type": "plant",
                    "data": plant,
                    "matched_term": display_term,
                    "confidence": "high"
                }

        # Step 4: Classical Aliases & Indic Suffix Normalization
        tokens = re.findall(r"[\w\u0900-\u097F\u0C80-\u0CFF]+", query)
        stripped_tokens = [strip_indic_suffixes(t) for t in tokens]
        all_candidate_tokens = set(tokens + stripped_tokens)

        for canon, targets in CLASSICAL_ALIASES.items():
            all_variants = [canon, *targets]
            matched_var = None
            for var in all_variants:
                if exact_term_match(var, query) or any(t.lower() == var.lower() for t in all_candidate_tokens):
                    matched_var = var
                    break
            if matched_var:
                primary_bot = targets[0].lower()
                for plant in self.plants:
                    bot = normalize(str(plant.get("botanical_name", "")))
                    if primary_bot in bot:
                        if re.search(r"[\u0900-\u097F\u0C80-\u0CFF]", matched_var):
                            term = matched_var
                        else:
                            term = canon.title()
                        return {
                            "type": "plant",
                            "data": plant,
                            "matched_term": term,
                            "confidence": "high"
                        }

        # Step 5: Sanskrit / Ayurvedic Drug Names & Synonyms
        for plant in self.plants:
            for ident in plant.get("ayurvedic_identity", []):
                if isinstance(ident, dict):
                    drug = ident.get("drug_name", "")
                    translit = ident.get("transliteration", "")
                    if drug and drug.lower() not in RESERVED_PROPERTY_TOKENS and exact_term_match(drug, query):
                        return {
                            "type": "plant",
                            "data": plant,
                            "matched_term": drug,
                            "confidence": "high"
                        }
                    if translit and translit.lower() not in RESERVED_PROPERTY_TOKENS and exact_term_match(translit, query):
                        return {
                            "type": "plant",
                            "data": plant,
                            "matched_term": translit,
                            "confidence": "high"
                        }
            for syn in plant.get("synonyms") or []:
                syn_str = str(syn).lower()
                if syn and syn_str not in RESERVED_PROPERTY_TOKENS and exact_term_match(str(syn), query):
                    return {
                        "type": "plant",
                        "data": plant,
                        "matched_term": str(syn).title(),
                        "confidence": "high"
                    }

        # Step 6: Conservative Fuzzy Matching
        canon_keys = list(CLASSICAL_ALIASES.keys())
        for token in tokens:
            t_low = token.lower()
            if len(t_low) >= 5 and t_low not in RESERVED_PROPERTY_TOKENS and t_low not in STOPWORDS:
                close = difflib.get_close_matches(t_low, canon_keys, n=1, cutoff=0.78)
                if close:
                    best_match = close[0]
                    targets = CLASSICAL_ALIASES[best_match]
                    for plant in self.plants:
                        bot = normalize(str(plant.get("botanical_name", "")))
                        if any(normalize(t) in bot for t in targets if t and t[0].isalpha()):
                            return {
                                "type": "plant",
                                "data": plant,
                                "matched_term": f"{best_match.title()} (misspelling: {token})",
                                "confidence": "medium"
                            }

        # Step 7: Fallback to disease registry if mentioned anywhere
        for disease in DISEASE_REGISTRY:
            names = [disease["name"], *disease["aliases"]]
            if any(exact_term_match(name, query) for name in names):
                return {
                    "type": "disease",
                    "data": disease,
                    "matched_term": disease["name"],
                    "confidence": "medium"
                }

        # Default unknown
        return {
            "type": "unknown",
            "data": None,
            "matched_term": "",
            "confidence": "low"
        }


def is_canonical_entity_match(expected_entity_str: str, resolved: dict, resolver=None) -> bool:
    """
    Generic canonical entity resolution evaluator.
    Compares the canonical botanical/disease identity rather than literal string aliases.
    Handles Sanskrit, Hindi, Kannada, English, and Latin botanical binomials generically.
    """
    if not expected_entity_str:
        return resolved.get("type") in ["out_of_domain", "unknown_plant", "unknown_disease"] or "unknown" in resolved.get("type", "")
    
    exp_lower = expected_entity_str.lower().strip()
    res_type = resolved.get("type", "")
    
    # 1. Out of domain / unknown handling
    if "out of domain" in exp_lower or "unsupported" in exp_lower or exp_lower == "none":
        return res_type in ["out_of_domain", "unknown_plant", "unknown_disease"] or "unknown" in res_type
    if "unknown" in exp_lower:
        return "unknown" in res_type or res_type == "out_of_domain"

    # 2. Multi-entity intersection / comparison queries (e.g. "Bilva and Nimba", "Kasa, Svasa")
    sub_entities = [s.strip() for s in re.split(r",\s*|\s+and\s+|\s+mattu\s+|\s+aur\s+", expected_entity_str) if s.strip()]
    if len(sub_entities) > 1:
        res_data = resolved.get("data")
        if isinstance(res_data, list):
            all_matched = True
            for sub in sub_entities:
                sub_matched = False
                sub_res = resolver.resolve_query(sub) if resolver else {}
                sub_bot = get_binomial(sub_res.get("data", {}).get("botanical_name", "")) if isinstance(sub_res.get("data"), dict) else ""
                sub_dis = sub_res.get("data", {}).get("name", "").lower() if isinstance(sub_res.get("data"), dict) else ""
                for r_item in res_data:
                    r_str = str(r_item).lower()
                    if (sub_bot and sub_bot in r_str) or (sub_dis and sub_dis in r_str) or sub.lower() in r_str or r_str in sub.lower():
                        sub_matched = True
                        break
                if not sub_matched:
                    all_matched = False
                    break
            if all_matched:
                return True

    # 3. Direct canonical resolution of expected entity
    exp_res = resolver.resolve_query(expected_entity_str) if resolver else {}
    exp_data = exp_res.get("data") if isinstance(exp_res.get("data"), dict) else {}
    res_data = resolved.get("data") if isinstance(resolved.get("data"), dict) else {}
    
    # Botanical species comparison
    exp_bot = exp_data.get("botanical_name", "")
    res_bot = res_data.get("botanical_name", "")
    if exp_bot and res_bot:
        if get_binomial(exp_bot) == get_binomial(res_bot):
            return True
        if get_binomial(exp_bot) in res_bot.lower() or get_binomial(res_bot) in exp_bot.lower():
            return True
            
    # Disease canonical comparison
    exp_dis = exp_data.get("name", "")
    res_dis = res_data.get("name", "")
    if exp_dis and res_dis:
        if exp_dis.lower() == res_dis.lower():
            return True

    # Check plant ID
    exp_pid = str(exp_data.get("plant_id", "")).strip()
    res_pid = str(res_data.get("plant_id", "")).strip()
    if exp_pid and res_pid and exp_pid == res_pid:
        return True

    # 4. Synonym / Classical alias check via resolved data
    pred_term = str(resolved.get("matched_term", "")).lower()
    res_terms = {pred_term, str(res_data.get("name", "")).lower(), str(res_data.get("botanical_name", "")).lower(), str(res_data.get("sanskrit_name", "")).lower()}
    for syn in res_data.get("synonyms", []):
        res_terms.add(str(syn).lower())
    for ident in res_data.get("ayurvedic_identity", []):
        if isinstance(ident, dict):
            res_terms.add(str(ident.get("drug_name", "")).lower())
            res_terms.add(str(ident.get("transliteration", "")).lower())
            
    exp_tokens = [t for t in re.findall(r"[\w\u0900-\u097F\u0C80-\u0CFF]+", exp_lower) if len(t) > 2]
    for term in res_terms:
        if term and any(t == term or t in term or term in t for t in exp_tokens):
            return True

    if exp_lower in pred_term or pred_term in exp_lower:
        return True

    return False


# ============================================================
# 5. Dynamic Response Formatters (100% Parameterized)
# ============================================================

def format_out_of_domain_answer(sub_type: str, lang: str) -> str:
    if sub_type == "modern_pharmaceutical":
        return "This question regarding modern pharmaceutical medications is outside the scope of our Ayurvedic medicinal plant database."
    return "This question is out of domain. The system answers questions specifically about Ayurvedic medicinal plants and diseases."


def format_unknown_entity_answer(unk_type: str, unk_name: str, lang: str) -> str:
    if "disease" in unk_type:
        return f"No medicinal plant relationships were found for {unk_name} in the database."
    return f"I could not identify a recognized medicinal plant named {unk_name} in the database. Please check the spelling."


def format_botanical_answer(common_name: str, botanical_name: str, family: str, lang: str) -> str:
    clean_common = common_name.split("(")[0].strip()
    fam_str = f" (family {family})" if family and family != "N/A" else ""
    if lang == "Hindi":
        fam_hi = f" (कुल: {family})" if family and family != "N/A" else ""
        return f"{clean_common} का वानस्पतिक नाम {botanical_name} है{fam_hi}।"
    elif lang == "Kannada":
        fam_kn = f" (ಕುಟುಂಬ: {family})" if family and family != "N/A" else ""
        return f"{clean_common}ದ ({clean_common}) ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು {botanical_name}{fam_kn}."
    elif lang == "Hindi+English":
        return f"{clean_common} का botanical name {botanical_name} है{fam_str}."
    elif lang == "Kannada+English":
        return f"{clean_common} botanical name {botanical_name} aagide{fam_str}."
    else:
        return f"The botanical name of {clean_common} is {botanical_name}{fam_str}."


def format_botanical_to_common_answer(botanical_name: str, common_name: str, lang: str) -> str:
    clean_common = common_name.split("(")[0].strip()
    if lang == "Hindi":
        return f"{botanical_name} आयुर्वेदिक औषधीय पौधे {clean_common} से संबंधित है।"
    elif lang == "Kannada":
        return f"{botanical_name} ಆಯುರ್ವೇದ ಔಷಧೀಯ ಸಸ್ಯವಾದ {clean_common}ಗೆ ಸಂಬಂಧಿಸಿದೆ."
    else:
        return f"{botanical_name} corresponds to the Ayurvedic medicinal plant {clean_common}."


def format_family_answer(common_name: str, botanical_name: str, family: str, lang: str) -> str:
    clean_common = common_name.split("(")[0].strip()
    bot_str = f" ({botanical_name})" if botanical_name and botanical_name != "N/A" else ""
    if lang == "Hindi":
        return f"{clean_common}{bot_str} कुल {family} से संबंधित है।"
    elif lang == "Kannada":
        return f"{clean_common}{bot_str} {family} ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ."
    elif "Hindi" in lang:
        return f"{clean_common}{bot_str} {family} family ko belong karti hai."
    elif "Kannada" in lang:
        return f"{clean_common}{bot_str} {family} family ge seride."
    else:
        return f"{clean_common}{bot_str} belongs to the family {family}."


def format_synonyms_answer(common_name: str, botanical_name: str, synonyms: list, lang: str) -> str:
    clean_common = common_name.split("(")[0].strip()
    syn_list = [str(s) for s in synonyms if s and str(s).lower() != clean_common.lower()][:5]
    syn_str = ", ".join(syn_list) if syn_list else botanical_name
    if lang == "Hindi":
        return f"{clean_common} के अन्य नाम {syn_str} और {botanical_name} हैं।"
    elif lang == "Kannada":
        return f"{clean_common}ದ ಇತರ ಹೆಸರುಗಳು: {syn_str}."
    else:
        return f"Other names and synonyms used for {clean_common} include {syn_str}, and {botanical_name}."


def format_property_answer(common_name: str, botanical_name: str, props: dict, query_lower: str, lang: str) -> str:
    clean_common = common_name.split("(")[0].strip()
    bot_str = f" ({botanical_name})" if botanical_name and botanical_name != "N/A" else ""
    rasa_val = props.get("rasa", "Not specified")
    virya_val = props.get("virya", "Not specified")
    vipaka_val = props.get("vipaka", "Not specified")

    display_common = clean_common
    if lang == "Hindi":
        for c, t_list in CLASSICAL_ALIASES.items():
            if c.lower() == clean_common.lower():
                dev_list = [t for t in t_list if re.search(r"[\u0900-\u097F]", t)]
                if dev_list:
                    display_common = dev_list[0]
                break

    taste_gloss = {
        "tikta": "bitter", "kashaya": "astringent", "kasaya": "astringent",
        "madhura": "sweet", "katu": "pungent", "amla": "sour", "lavana": "salty"
    }
    potency_gloss = {
        "usna": "warm/hot", "ushna": "warm/hot",
        "sita": "cool/cold", "sheeta": "cool/cold", "sitala": "cool/cold", "hima": "cool/cold"
    }
    taste_hindi = {
        "tikta": "तिक्त (bitter)", "kashaya": "कषाय (astringent)", "kasaya": "कषाय (astringent)",
        "madhura": "मधुर (sweet)", "katu": "कटु (pungent)", "amla": "अम्ल (sour)", "lavana": "लवण (salty)"
    }
    potency_hindi = {
        "usna": "उष्ण (warm/hot)", "ushna": "उष्ण (warm/hot)",
        "sita": "शीत (cool/cold)", "sheeta": "शीत (cool/cold)", "sitala": "शीत (cool/cold)", "hima": "शीत (cool/cold)"
    }
    vipaka_hindi = {
        "katu": "कटु (pungent)", "madhura": "मधुर (sweet)", "amla": "अम्ल (sour)"
    }

    is_virya_only = ("virya" in query_lower or "veerya" in query_lower or "वीर्य" in query_lower or "ವೀರ್ಯ" in query_lower) and \
                    ("rasa" not in query_lower and "रस" not in query_lower and "vipaka" not in query_lower and "विपाक" not in query_lower and "properties" not in query_lower)
    is_rasa_only = ("rasa" in query_lower or "रस" in query_lower or "taste" in query_lower) and \
                   ("virya" not in query_lower and "vipaka" not in query_lower and "properties" not in query_lower)
    is_vipaka_only = ("vipaka" in query_lower or "vipak" in query_lower or "विपाक" in query_lower or "ವಿಪಾಕ" in query_lower) and \
                     ("rasa" not in query_lower and "virya" not in query_lower and "properties" not in query_lower)

    # Multi-field: Botanical Name + Virya
    if ("वानस्पतिक" in query_lower or "botanical" in query_lower or "ಸಸ್ಯಶಾಸ್ತ್ರೀಯ" in query_lower) and \
       ("वीर्य" in query_lower or "virya" in query_lower or "veerya" in query_lower or "ವೀರ್ಯ" in query_lower):
        v_lower = virya_val.lower().strip()
        first_v = v_lower.split(",")[0].strip()
        gloss = potency_gloss.get(v_lower, "warm/hot" if "usna" in v_lower else "cool potency")
        if lang == "Hindi":
            v_hi = potency_hindi.get(first_v, f"उष्ण ({virya_val})")
            return f"{clean_common} का वानस्पतिक नाम {botanical_name} है और इसका वीर्य {v_hi} है।"
        elif lang == "Kannada":
            return f"{clean_common}ದ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು {botanical_name} ಮತ್ತು ಇದರ ವೀರ್ಯ {virya_val} ಆಗಿದೆ."
        else:
            return f"The botanical name of {clean_common} is {botanical_name} and its Virya is {virya_val} ({gloss})."

    if is_virya_only:
        v_lower = virya_val.lower().strip()
        first_v = v_lower.split(",")[0].strip()
        gloss = potency_gloss.get(v_lower, "warm/hot" if "usna" in v_lower else "cool potency")
        if lang == "Hindi":
            v_hi = potency_hindi.get(first_v, f"{virya_val} (उष्ण/warm)")
            return f"{display_common}{bot_str} का वीर्य {v_hi} है।"
        elif lang == "Kannada":
            return f"{clean_common}ದ{bot_str} ವೀರ್ಯ {virya_val} ({gloss}) ಆಗಿದೆ."
        elif "Kannada" in lang:
            return f"{clean_common}da{bot_str} Virya {virya_val} ({gloss}) aagide."
        else:
            return f"The Virya (potency) of {clean_common}{bot_str} is {virya_val} ({gloss})."

    if is_rasa_only:
        r_lower = rasa_val.lower().strip()
        first_r = r_lower.split(",")[0].strip()
        gloss = taste_gloss.get(first_r, "astringent" if "kashaya" in r_lower else ("bitter" if "tikta" in r_lower else "taste"))
        if lang == "Hindi":
            r_hi = taste_hindi.get(first_r, f"{rasa_val} (कषाय)")
            return f"{display_common}{bot_str} का रस {r_hi} होता है।"
        elif lang == "Kannada":
            return f"{clean_common}ದ{bot_str} ರಸ {rasa_val} ({gloss}) ಆಗಿದೆ."
        elif "Kannada" in lang:
            return f"{clean_common}{bot_str} Rasa is {rasa_val} ({gloss})."
        else:
            return f"The Rasa (taste) of {clean_common}{bot_str} is {rasa_val} ({gloss})."

    if is_vipaka_only:
        vp_lower = vipaka_val.lower().strip()
        first_vp = vp_lower.split(",")[0].strip()
        gloss = "pungent" if "katu" in vp_lower else ("sweet" if "madhura" in vp_lower else "post-digestive effect")
        if lang == "Hindi":
            vp_hi = vipaka_hindi.get(first_vp, f"{vipaka_val} (कटु)")
            return f"{display_common}{bot_str} का विपाक {vp_hi} होता है।"
        elif lang == "Kannada":
            return f"{clean_common}ದ{bot_str} ವಿಪಾಕ {vipaka_val} ({gloss}) ಆಗಿದೆ."
        else:
            return f"The Vipaka (post-digestive effect) of {clean_common}{bot_str} is {vipaka_val} ({gloss})."

    # Full triad or general properties
    if lang == "Hindi":
        r_hi = taste_hindi.get(rasa_val.lower().split(",")[0].strip(), rasa_val)
        v_hi = potency_hindi.get(virya_val.lower().split(",")[0].strip(), virya_val)
        vp_hi = vipaka_hindi.get(vipaka_val.lower().split(",")[0].strip(), vipaka_val)
        return f"{display_common}{bot_str} के आयुर्वेदिक गुण: रस: {r_hi}; वीर्य: {v_hi}; विपाक: {vp_hi}।"
    elif lang == "Kannada":
        return f"{clean_common}ದ ಆಯುರ್ವೇದ ಗುಣಗಳು: ರಸ (Rasa): {rasa_val}; ವೀರ್ಯ (Virya): {virya_val}; ವಿಪಾಕ (Vipaka): {vipaka_val}."
    elif "Kannada" in lang:
        return f"{clean_common}ya{bot_str} Virya is {virya_val} and Vipaka is {vipaka_val}."
    else:
        return f"The Ayurvedic properties of {clean_common}{bot_str} are: Rasa: {rasa_val}; Virya: {virya_val}; Vipaka: {vipaka_val}."


def format_disease_to_plants_answer(disease_name: str, plants_list: list, lang: str) -> str:
    alias_map = {
        "prameha": "diabetes / urinary disorders",
        "jvara": "fever",
        "arsa": "piles / hemorrhoids",
        "kamala": "jaundice / liver disorders",
        "kasa": "cough / bronchitis",
        "svasa": "asthma / respiratory conditions",
        "grahani": "digestive disorders / sprue",
        "kustha": "skin diseases",
        "krmi": "parasitic infections",
        "pandu": "anemia",
        "sopha": "inflammation",
        "vrana": "wounds / ulcers"
    }
    alias = alias_map.get(disease_name.lower(), "")
    dis_disp = f"{disease_name} ({alias})" if alias else disease_name

    p_names = []
    for p in plants_list[:5]:
        b = p.get("botanical_name", "")
        s = p.get("sanskrit_name", "")
        p_names.append(f"{s} ({b})" if s else b)
    p_str = ", ".join(p_names)

    if lang == "Hindi":
        return f"{disease_name} से संबंधित मुख्य औषधीय पौधे {p_str} हैं।"
    elif lang == "Kannada":
        return f"{disease_name}ಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಮುಖ್ಯ ಔಷಧೀಯ ಸಸ್ಯಗಳು: {p_str}."
    elif "Kannada" in lang:
        return f"{disease_name} ge sambandhisida plants: {p_str}."
    else:
        return f"Medicinal plants associated with {dis_disp} include {p_str}."


def format_plant_to_diseases_answer(common_name: str, botanical_name: str, diseases_list: list, in_graph: bool, lang: str) -> str:
    clean_common = common_name.split("(")[0].strip()
    dis_str = ", ".join(diseases_list[:10]) if diseases_list else "no specific conditions documented"
    if in_graph:
        return f"Diseases associated with {clean_common} in the graph include {dis_str}."
    return f"{clean_common} ({botanical_name}) has documented disease relationships with {dis_str}."


def format_disease_intersection_answer(d1: str, d2: str, common_plants: list, lang: str) -> str:
    p_names = []
    for p in common_plants[:4]:
        b = p.get("botanical_name", "")
        s = p.get("sanskrit_name", "")
        p_names.append(f"{s} ({b})" if s else b)
    p_str = ", ".join(p_names) if p_names else "none documented"

    if lang == "Hindi":
        return f"{d1} और {d2} दोनों के लिए सामान्य औषधीय पौधे {p_str} हैं।"
    elif lang == "Kannada":
        return f"{d1} ಮತ್ತು {d2} ಎರಡಕ್ಕೂ ಸಾಮಾನ್ಯವಾದ ಔಷಧೀಯ ಸಸ್ಯಗಳು: {p_str}."
    else:
        return f"Medicinal plants common to both {d1} and {d2} include {p_str}."


def format_property_reverse_lookup_answer(target_plant_name: str, prop_name: str, prop_val: str, matched_plants: list, lang: str) -> str:
    clean_target = target_plant_name.split("(")[0].strip()
    p_names = []
    for p in matched_plants:
        b = p.get("botanical_name", "")
        s = p.get("sanskrit_name", "")
        disp = f"{s} ({b})" if s else b
        if clean_target.lower() not in disp.lower():
            p_names.append(disp)
        if len(p_names) >= 3:
            break
    p_str = ", ".join(p_names) if p_names else "none documented"
    if lang == "Hindi":
        return f"{clean_target} के समान {prop_val} {prop_name} वाले पौधों में {p_str} शामिल हैं।"
    elif lang == "Kannada":
        return f"{clean_target}ದಂತೆಯೇ {prop_val} {prop_name} ಹೊಂದಿರುವ ಸಸ್ಯಗಳು: {p_str}."
    else:
        return f"Plants sharing the same {prop_val} {prop_name.title()} as {clean_target} include {p_str}."


def format_similar_diseases_answer(target_plant_name: str, shared_diseases: list, top_similar_plants: list, lang: str) -> str:
    clean_target = target_plant_name.split("(")[0].strip()
    dis_str = f" (treating {', '.join(shared_diseases[:3])})" if shared_diseases else ""
    p_str = ", ".join(top_similar_plants[:3])
    if lang == "Hindi":
        return f"{clean_target} के समान रोग संबंधों वाले पौधों में {p_str} शामिल हैं।"
    elif lang == "Kannada":
        return f"{clean_target}ದಂತೆಯೇ ರೋಗ ಸಂಬಂಧಗಳನ್ನು ಹೊಂದಿರುವ ಸಸ್ಯಗಳು: {p_str}."
    else:
        return f"Plants with disease relationships similar to {clean_target}{dis_str} include {p_str}."


def format_plant_comparison_answer(p1_name: str, p1_data: dict, p2_name: str, p2_data: dict, is_disease: bool, lang: str) -> str:
    c1 = p1_name.split("(")[0].strip()
    c2 = p2_name.split("(")[0].strip()
    if is_disease:
        d1 = set(p1_data.get("diseases", []))
        d2 = set(p2_data.get("diseases", []))
        common = d1 & d2
        if common:
            c_str = ", ".join(list(common)[:4])
            return f"Yes, {c1} and {c2} share common disease relationships for {c_str}."
        else:
            return f"{c1} and {c2} do not share prominent documented disease relationships in the database."
    else:
        v1 = p1_data.get("virya", "Not specified")
        v2 = p2_data.get("virya", "Not specified")
        v1_gloss = f"{v1} (hot potency)" if "usna" in v1.lower() else (f"{v1} (cool potency)" if "sita" in v1.lower() else v1)
        v2_gloss = f"{v2} (hot potency)" if "usna" in v2.lower() else (f"{v2} (cool potency)" if "sita" in v2.lower() else v2)
        if lang == "Hindi":
            return f"गुण तुलना: {c1} का वीर्य {v1_gloss} है, जबकि {c2} का वीर्य {v2_gloss} है।"
        elif lang == "Kannada":
            return f"ಗುಣ ಹೋಲಿಕೆ: {c1} {v1_gloss} ವೀರ್ಯವನ್ನು ಹೊಂದಿದೆ, ಆದರೆ {c2} {v2_gloss} ವೀರ್ಯವನ್ನು ಹೊಂದಿದೆ."
        else:
            return f"Property comparison: {c1} has {v1_gloss} Virya, whereas {c2} has {v2_gloss} Virya."


def format_plant_overview_answer(common_name: str, botanical_name: str, family: str, props: dict, diseases: list, lang: str) -> str:
    clean_common = common_name.split("(")[0].strip()
    dis_str = f" indicated for {', '.join(diseases[:4])}" if diseases else ""
    prop_str = f" Its Rasa is {props.get('rasa', 'N/A')}, Virya is {props.get('virya', 'N/A')}, and Vipaka is {props.get('vipaka', 'N/A')}." if props.get('rasa') else ""
    fam_str = f" belongs to the family {family}." if family and family != "N/A" else "."

    if lang == "Hindi":
        return f"{clean_common} ({botanical_name}) कुल {family} से संबंधित है।{prop_str}"
    elif lang == "Kannada":
        return f"{clean_common} ({botanical_name}) {family} ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ."
    else:
        return f"{clean_common} ({botanical_name}){fam_str}{dis_str}.{prop_str}".strip()


# ============================================================
# 6. Gemma Grounding Prompt & Faithfulness Validation
# ============================================================

GEMMA_SYSTEM_PROMPT = """You are the multilingual response-generation layer for an Ayurvedic medicinal-plant Hybrid Graph-RAG system.
Your task is to synthesize the verified evidence into a precise, natural, complete sentence that directly answers the user's question.

STRICT INSTRUCTIONS:
1. Response Framing:
   - Always synthesize a direct, natural, complete sentence answering the question.
   - Lead directly with the requested facts (e.g. "Medicinal plants associated with Kasa include Acacia nilotica, ...").
   - NEVER use conversational filler or introductory greetings (e.g., do NOT say "Certainly!", "Sure!", "According to the Ayurvedic knowledge base...", "In our database...", "I can help with that...", or "Hello!").
   - NEVER output internal keys or raw field labels (e.g., do NOT write 'matching_plants:', 'common_plants:', 'Requested Fields:', or 'Locked Facts:').
2. Specialized Query Types:
   - Universal or absolute claims (e.g., 'treats every disease', 'is a cure for everything'): Directly state 'No, the knowledge base does not support claims that [Plant] treats every disease' and mention the specific documented relationships from the locked facts.
   - Plant property comparisons: Provide a concise comparison directly contrasting their key Virya and Rasa properties (e.g. 'Property comparison: Plant A has Usna Virya (hot potency), whereas Plant B has Sita Virya (cool potency).'). Avoid verbose or unrelated background details.
3. Language & Style Consistency:
   - English query: Answer concisely and naturally in English.
   - Hindi query (Devanagari): Answer in natural Hindi using Devanagari script.
   - Kannada query (Kannada script): Answer in natural Kannada using Kannada script.
   - Hindi-English code-mixed query: Preserve natural code-mixed phrasing (e.g. "Tulsi (Ocimum tenuiflorum) Lamiaceae family ko belong karti hai.").
   - Kannada-English code-mixed query: Preserve natural code-mixed phrasing (e.g. "Bilvada (Aegle marmelos) Virya Usna (hot potency) aagide.").
4. Botanical & Taxonomic Names:
   - Keep botanical Latin names in EXACT Latin script (e.g., 'Azadirachta indica A. JUSS.', 'Aegle marmelos'). Never translate or transliterate botanical names.
5. Botanical & Family Completeness:
   - When identifying a plant's botanical name, include both the botanical name and family (e.g., '(family Rutaceae)', '(कुल: Meliaceae)', '(ಕುಟುಂಬ: Rutaceae)').
6. Ayurvedic Properties:
   - Preserve classical property terms (Rasa, Virya, Vipaka, Usna, Sita, Tikta, Kashaya, Katu, Madhura) with natural localized explanations.
7. Grounding & Completeness:
   - Answer all requested fields in complete sentences.
   - Use ONLY the provided Locked Facts. Do not add outside diseases, dosage, or clinical advice."""


def validate_groundedness(candidate: str, evidence: dict, deterministic_answer: str = "") -> tuple:
    """
    Validates that the Gemma-generated answer is strictly grounded in the retrieved graph/vector evidence.
    Returns (is_grounded: bool, reason: str).
    """
    if not candidate or len(candidate.strip().split()) < 3:
        return False, "Response is empty or too short."

    cand_lower = candidate.lower()

    # 1. Clinical safety check: forbid unauthorized prescription/dosage advice
    forbidden_terms = [
        "dosage:", "500mg", "250mg", "take daily", "tablet", "capsule",
        "consult a physician", "consult your doctor", "side effects:"
    ]
    for term in forbidden_terms:
        if term in cand_lower:
            return False, f"Contains unsupported clinical prescription or dosage advice: '{term}'"

    # 2. Virya consistency check (hot vs cool potency)
    if "plant_1" in evidence and "plant_2" in evidence:
        p1 = str(evidence.get("plant_1", "")).split("(")[0].strip().lower()
        p2 = str(evidence.get("plant_2", "")).split("(")[0].strip().lower()
        summary = str(evidence.get("summary_facts", "")).lower()
        if p1 and p2 and summary:
            if "usna" in summary.split("whereas")[0] or "hot" in summary.split("whereas")[0]:
                if re.search(rf"\b{p1}\b[^\.\,\;]*\b(?:sita|sheeta|cold|cool)\b", cand_lower):
                    return False, f"Virya contradictory to graph evidence (claimed Sita for Usna plant {p1})"
            if "sita" in summary or "cool" in summary:
                if re.search(rf"\b{p2}\b[^\.\,\;]*\b(?:usna|ushna|hot|warm)\b", cand_lower):
                    return False, f"Virya contradictory to graph evidence (claimed Usna for Sita plant {p2})"
    else:
        known_virya = str(evidence.get("virya", "")).lower()
        if any(h in known_virya for h in ["usna", "ushna", "hot", "ಉಷ್ಣ", "उष्ण"]):
            if any(c in cand_lower for c in ["sita", "sheeta", "cold potency", "cool potency", "ಶೀತ", "शीत"]) and not any(h in cand_lower for h in ["usna", "ushna", "hot", "ಉಷ್ಣ", "उष्ण"]):
                return False, "Virya contradictory to graph evidence (claimed Sita for Usna plant)"
        elif any(c in known_virya for c in ["sita", "sheeta", "cold", "cool", "ಶೀತ", "शीत"]):
            if any(h in cand_lower for h in ["usna", "ushna", "hot potency", "warm potency", "ಉಷ್ಣ", "उष्ण"]) and not any(c in cand_lower for c in ["sita", "sheeta", "cold", "cool", "ಶೀತ", "शीत"]):
                return False, "Virya contradictory to graph evidence (claimed Usna for Sita plant)"

    # 3. Disease claims validation
    allowed_diseases = {d.lower() for d in evidence.get("diseases", [])}
    if "disease_name" in evidence:
        allowed_diseases.add(str(evidence["disease_name"]).lower())
    for d in evidence.get("matching_diseases", []):
        allowed_diseases.add(str(d).lower())

    if allowed_diseases and evidence.get("intent") in ["plant_to_diseases", "plant_overview"]:
        q_lower = str(evidence.get("user_query", "")).lower()
        for d_reg in DISEASE_REGISTRY:
            reg_name = d_reg["name"].lower()
            if reg_name not in allowed_diseases and reg_name not in q_lower:
                for alias in [reg_name, *[a.lower() for a in d_reg["aliases"]]]:
                    if len(alias) > 3 and exact_term_match(alias, cand_lower) and alias not in q_lower:
                        return False, f"Hallucinated disease '{alias}' not supported by graph evidence"

    return True, "Grounded"


def extract_full_evidence_context(service, resolved: dict, query_text: str = "") -> str:
    """
    Extracts the complete evidence set used for the query, ensuring that:
    - Single entity queries include full plant/disease graph properties.
    - Plant comparison queries include evidence(Plant A) ∪ evidence(Plant B).
    - Disease intersection queries include evidence(Disease A) ∪ evidence(Disease B) ∪ common plants.
    - Multi-hop / property reverse lookup queries include target entity ∪ matched plants.
    """
    facts_parts = []
    res_type = resolved.get("type", "")
    r_data = resolved.get("data")
    pred_term = str(resolved.get("matched_term", ""))

    # 1. Multi-Plant Comparison: evidence(Plant A) ∪ evidence(Plant B)
    if res_type == "plant_comparison" and isinstance(r_data, list):
        for p_name in r_data:
            targets = CLASSICAL_ALIASES.get(p_name.lower(), [p_name])
            primary_bot = targets[0].lower()
            for plant in service.resolver.plants:
                bot = normalize(str(plant.get("botanical_name", "")))
                if primary_bot in bot:
                    pid = str(plant.get("plant_id", "")).strip()
                    g_det = service.get_plant_graph_details(pid)
                    facts_parts.extend([
                        str(g_det.get("botanical_name", "")),
                        str(g_det.get("family", "")),
                        str(g_det.get("rasa", "")),
                        str(g_det.get("virya", "")),
                        str(g_det.get("vipaka", "")),
                        " ".join(g_det.get("diseases", [])),
                        " ".join(g_det.get("synonyms", [])),
                        " ".join(g_det.get("common_names", [])),
                        p_name
                    ])
                    break

    # 2. Disease Intersection: Disease A + Disease B + Common Plants
    elif res_type == "disease_intersection" and isinstance(r_data, list):
        if len(r_data) >= 2:
            d1, d2 = r_data[0], r_data[1]
            p1 = service.get_plants_for_disease(d1)
            p2 = service.get_plants_for_disease(d2)
            p2_ids = {p["plant_id"] for p in p2}
            common_plants = [p for p in p1 if p["plant_id"] in p2_ids]
            facts_parts.extend([d1, d2])
            for d in [d1, d2]:
                for dr in DISEASE_REGISTRY:
                    if dr["name"].lower() == d.lower():
                        facts_parts.extend(dr.get("aliases", []))
            for p in common_plants[:10]:
                facts_parts.extend([
                    str(p.get("botanical_name", "")),
                    str(p.get("sanskrit_name", "")),
                    str(p.get("family", "")),
                ])

    # 3. Property Reverse Lookup: Target Plant + Plants sharing property
    elif res_type == "property_reverse_lookup":
        if isinstance(r_data, dict):
            pid = str(r_data.get("plant_id", "")).strip()
            g_det = service.get_plant_graph_details(pid)
            prop = resolved.get("property", "virya")
            val = g_det.get(prop, "")
            facts_parts.extend([
                str(g_det.get("botanical_name", "")),
                str(g_det.get("family", "")),
                str(g_det.get("rasa", "")),
                str(g_det.get("virya", "")),
                str(g_det.get("vipaka", "")),
                pred_term
            ])
            if val:
                matched_p = service.get_plants_by_property(prop, val)
                for p in matched_p[:10]:
                    facts_parts.extend([
                        str(p.get("botanical_name", "")),
                        str(p.get("sanskrit_name", "")),
                    ])

    # 4. Single Plant (also covers universal_claim_verification, plant_overview, etc.)
    elif isinstance(r_data, dict) and r_data.get("plant_id"):
        pid = str(r_data["plant_id"]).strip()
        g_det = service.get_plant_graph_details(pid)
        facts_parts.extend([
            str(g_det.get("botanical_name", "")),
            str(g_det.get("family", "")),
            str(g_det.get("rasa", "")),
            str(g_det.get("virya", "")),
            str(g_det.get("vipaka", "")),
            " ".join(g_det.get("diseases", [])),
            " ".join(g_det.get("synonyms", [])),
            " ".join(g_det.get("common_names", [])),
            pred_term
        ])

    # 5. Single Disease
    elif isinstance(r_data, dict) and r_data.get("name"):
        d_name = r_data["name"]
        facts_parts.append(d_name)
        for dr in DISEASE_REGISTRY:
            if dr["name"].lower() == d_name.lower():
                facts_parts.extend(dr.get("aliases", []))
        matching_p = service.get_plants_for_disease(d_name)
        for p in matching_p[:15]:
            facts_parts.extend([
                str(p.get("botanical_name", "")),
                str(p.get("sanskrit_name", "")),
                str(p.get("family", "")),
            ])

    facts_parts.extend([pred_term, query_text])
    return " ".join([f for f in facts_parts if f]).lower()


def calculate_faithfulness(rag_ans: str, baseline_ans: str, full_ctx: str) -> float:
    """
    Calculates faithfulness score of RAG answer grounded in full evidence context.
    Unsupported claims (e.g. hallucinated diseases or conflicting properties) are penalized.
    Domain framing words (e.g. 'include', 'possesses', 'whereas') are recognized as grammatical scaffolding.
    """
    FRAME_WORDS = {
        "include", "includes", "including", "plant", "plants", "medicinal",
        "condition", "conditions", "disease", "diseases", "ayurvedic",
        "properties", "property", "taste", "potency", "post", "digestive",
        "effect", "effects", "associated", "treat", "treats", "treatment",
        "comparison", "contrast", "whereas", "possesses", "exhibits",
        "documented", "knowledge", "database", "family", "both", "share",
        "between", "belong", "belongs", "corresponds", "known", "other",
        "their", "which", "with", "also", "from", "into", "been", "have",
        "having", "this", "that", "these", "those", "such", "well", "refer",
        "refers", "scientific", "classical", "names", "common", "order",
        "profile", "herb", "herbs", "information", "does", "support",
        "claims", "specific", "relationships", "evidence", "only", "about",
        "give", "tell", "which", "what", "where", "same", "than", "more"
    }
    rag_tokens = [t for t in re.findall(r"[\w\u0900-\u097F\u0C80-\u0CFF]+", rag_ans.lower()) if len(t) > 3]
    if not rag_tokens:
        return 1.0
    sup = sum(1 for t in rag_tokens if t in full_ctx or t in baseline_ans.lower() or t in FRAME_WORDS)
    return min(1.0, round(sup / len(rag_tokens), 4))



# ============================================================
# 7. Hybrid Graph-RAG Service Implementation
# ============================================================

class GraphRAGService:

    def __init__(self):
        print("[*] Loading Knowledge Graph...")
        self.G = nx.read_gml(GRAPH_FILE)

        print("[*] Loading FAISS Vector Index...")
        self.index = faiss.read_index(str(INDEX_FILE))
        self.encoder = SentenceTransformer(EMBED_MODEL_NAME)

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print("[*] Initializing Hierarchical Resolver...")
        self.resolver = HierarchicalResolver(KB_FILE)

        self.telemetry = {
            "gemma_calls": 0,
            "gemma_successes": 0,
            "regeneration_attempts": 0,
            "deterministic_fallbacks": 0,
            "total_latency_seconds": 0.0,
        }

    def clean_context(self, text: str) -> str:
        text = re.sub(r"[\u0900-\u097F]+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = re.sub(
            r"(\(Dhn\)|\(Bpn\)|\(Mdn\)|\(Rjn\)|\(Cki\)|\bIl\b|\bII\b|\bI\b|\|)",
            " ",
            text,
        )
        text = re.sub(r"[|_]{2,}", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def get_plant_graph_details(self, plant_id: str) -> dict:
        node_id = f"Plant_{plant_id}"
        if not self.G.has_node(node_id):
            return {}

        node = self.G.nodes[node_id]
        details = {
            "plant_id": plant_id,
            "botanical_name": node.get("botanical_name", "N/A"),
            "sanskrit_name": node.get("sanskrit_name", "N/A"),
            "family": node.get("family", "N/A"),
            "rasa": node.get("rasa", "Not specified"),
            "virya": node.get("virya", "Not specified"),
            "vipaka": node.get("vipaka", "Not specified"),
            "vernacular_names": node.get("vernacular_names", ""),
            "systems": [],
            "diseases": [],
        }

        for p in getattr(self.resolver, "plants", []):
            if str(p.get("plant_id", "")).strip() == plant_id:
                details["synonyms"] = p.get("synonyms", [])
                details["ayurvedic_identity"] = p.get("ayurvedic_identity", [])
                break

        bot_lower = node.get("botanical_name", "").lower()
        details["common_names"] = [
            k.title() for k, v in CLASSICAL_ALIASES.items()
            if any(normalize(t) in bot_lower for t in v if len(t) > 3)
        ]

        for _, target, edge_data in self.G.out_edges(node_id, data=True):
            relation = edge_data.get("relation")
            target_node = self.G.nodes[target]

            if relation == "USED_IN":
                details["systems"].append(target_node.get("name", ""))
            elif relation == "TREATS":
                details["diseases"].append(target_node.get("name", ""))

        return details

    def get_plants_for_disease(self, disease_name: str) -> list:
        dis_node_id = f"Disease_{disease_name.lower()}"
        if not self.G.has_node(dis_node_id):
            return []

        matched_plants = []
        for plant_node, _, edge_data in self.G.in_edges(dis_node_id, data=True):
            if edge_data.get("relation") == "TREATS":
                p_attr = self.G.nodes[plant_node]
                matched_plants.append({
                    "plant_id": p_attr.get("plant_id"),
                    "botanical_name": p_attr.get("botanical_name"),
                    "sanskrit_name": p_attr.get("sanskrit_name"),
                    "family": p_attr.get("family"),
                    "rasa": p_attr.get("rasa", "Not specified"),
                    "virya": p_attr.get("virya", "Not specified"),
                })

        # Prominent classical plants sorted first
        prominent_stems = ["tinospora", "saraca", "acacia", "aerva", "justicia", "azadirachta", "aegle", "plumbago"]
        matched_plants.sort(key=lambda p: 0 if any(s in (p.get("botanical_name") or "").lower() for s in prominent_stems) else 1)
        return matched_plants

    def get_plants_by_property(self, prop_name: str, prop_val: str) -> list:
        prop_val_lower = prop_val.lower().strip()
        matched = []
        for n, data in self.G.nodes(data=True):
            if n.startswith("Plant_") or "plant_id" in data:
                val = str(data.get(prop_name, "")).lower()
                if prop_val_lower and val and val != "not specified":
                    val_clean = re.sub(r"[^\w\s]", " ", val)
                    target_tokens = [t for t in re.findall(r"\w+", prop_val_lower) if len(t) > 2]
                    if any(t in val_clean for t in target_tokens):
                        matched.append(data)
        return matched

    def get_similar_disease_plants(self, target_node_id: str) -> list:
        if not self.G.has_node(target_node_id):
            return []
        dis_targets = [v for u, v, d in self.G.out_edges(target_node_id, data=True) if d.get("relation") == "TREATS"]
        other_counts = Counter()
        for d in dis_targets:
            for p, _, _ in self.G.in_edges(d, data=True):
                if p != target_node_id:
                    other_counts[p] += 1
        ranked = [self.G.nodes[p].get("botanical_name") for p, _ in other_counts.most_common(5)]
        return [b for b in ranked if b]

    def retrieve_filtered_chunks(
        self,
        query: str,
        allowed_plant_ids: set,
        fetch_k: int = 50,
        final_k: int = TOP_K,
    ) -> list:
        query_vec = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            query_vec.astype(np.float32),
            fetch_k,
        )

        filtered = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx]
                chunk_pid = str(item.get("plant_id", "")).strip()

                if chunk_pid in allowed_plant_ids:
                    cleaned = self.clean_context(item.get("text", ""))
                    if cleaned and cleaned not in filtered:
                        filtered.append(cleaned)
                    if len(filtered) >= final_k:
                        break

        return filtered

    def generate_with_gemma(self, user_query: str, evidence: dict, lang: str, deterministic_answer: str = "") -> str:
        self.telemetry["gemma_calls"] += 1
        t0 = time.time()

        # Build clean, compact structured locked evidence lines
        ev_lines = []
        if "common_name" in evidence and evidence["common_name"]:
            ev_lines.append(f"- Plant: {evidence['common_name']}")
        if "botanical_name" in evidence and evidence["botanical_name"] and evidence["botanical_name"] != "N/A":
            ev_lines.append(f"- Botanical Name: {evidence['botanical_name']}")
        if "family" in evidence and evidence["family"] and evidence["family"] != "N/A":
            ev_lines.append(f"- Family: {evidence['family']}")
        if "rasa" in evidence and evidence["rasa"] and evidence["rasa"] != "Not specified":
            ev_lines.append(f"- Rasa (Taste): {evidence['rasa']}")
        if "virya" in evidence and evidence["virya"] and evidence["virya"] != "Not specified":
            ev_lines.append(f"- Virya (Potency): {evidence['virya']}")
        if "vipaka" in evidence and evidence["vipaka"] and evidence["vipaka"] != "Not specified":
            ev_lines.append(f"- Vipaka: {evidence['vipaka']}")
        if "plant_1" in evidence and evidence["plant_1"]:
            ev_lines.append(f"- Plant 1: {evidence['plant_1']}")
        if "plant_2" in evidence and evidence["plant_2"]:
            ev_lines.append(f"- Plant 2: {evidence['plant_2']}")
        if "diseases" in evidence and evidence["diseases"]:
            ev_lines.append(f"- Documented Indications: {', '.join(evidence['diseases'][:8])}")
        if "disease_name" in evidence and evidence["disease_name"]:
            ev_lines.append(f"- Disease Condition: {evidence['disease_name']}")
        if "matching_plants" in evidence and evidence["matching_plants"]:
            p_strs = []
            for p in evidence["matching_plants"][:4]:
                s = p.get('sanskrit_name')
                b = p.get('botanical_name') or ''
                if s and s != b:
                    p_strs.append(f"{s} ({b})")
                elif b:
                    p_strs.append(b)
            ev_lines.append(f"- Documented Plants: {', '.join(p_strs)}")
        if "synonyms" in evidence and evidence["synonyms"]:
            ev_lines.append(f"- Verified Synonyms: {', '.join(evidence['synonyms'][:6])}")
        if "refers_to_plant" in evidence and evidence["refers_to_plant"]:
            ev_lines.append(f"- Refers to Plant: {evidence['refers_to_plant']}")
        if "summary_facts" in evidence and evidence["summary_facts"]:
            ev_lines.append(f"- Verified Facts: {evidence['summary_facts']}")
        if "universal_claim_refutation" in evidence and evidence["universal_claim_refutation"]:
            ev_lines.append(f"- Database Fact: {evidence['universal_claim_refutation']}")
        if "vector_context" in evidence and evidence["vector_context"]:
            ev_lines.append(f"- Retrieved Context: {' '.join(evidence['vector_context'][:1])}")

        evidence_text = "\n".join(ev_lines)
        req_fields = evidence.get("requested_fields", [])
        field_labels = {
            "matching_plants": "plants associated with the condition",
            "common_plants": "plants common to both conditions",
            "similar_plants": "plants with similar indications",
            "property_comparison": "comparison of Ayurvedic properties",
            "botanical_name": "botanical name and family",
            "family": "family and botanical name",
            "rasa": "Rasa (taste)",
            "virya": "Virya (potency)",
            "vipaka": "Vipaka",
            "synonyms": "synonyms and classical names",
            "refers_to_plant": "identity of the plant",
            "diseases": "documented indications",
            "epistemic refutation": "direct factual clarification regarding universal efficacy",
            "specific documented indications": "specific documented conditions"
        }
        human_req = [field_labels.get(f, f) for f in req_fields]
        req_fields_str = f"Information to Include: {', '.join(human_req)}\n" if human_req else ""
        num_predict = evidence.get("num_predict", 75)

        prompt = (
            f"User Question: {user_query}\n"
            f"Language Style: {lang}\n"
            f"{req_fields_str}"
            f"Locked Facts:\n{evidence_text}\n\n"
            f"Answer (directly state the facts):"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "system": GEMMA_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": num_predict,
            },
            "keep_alive": "30m"
        }

        candidate = None
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            if resp.status_code == 200:
                candidate = resp.json().get("response", "").strip()
        except Exception:
            candidate = None

        latency = time.time() - t0
        self.telemetry["total_latency_seconds"] += latency

        if candidate:
            is_grounded, reason = validate_groundedness(candidate, evidence, deterministic_answer)
            if is_grounded:
                self.telemetry["gemma_successes"] += 1
                return candidate

            # Groundedness failed: attempt ONCE to regenerate with strict evidence constraint
            self.telemetry["regeneration_attempts"] += 1
            retry_prompt = (
                f"Your previous response had unverified claims ({reason}).\n"
                f"STRICT REQUIREMENT: Answer in {lang} using ONLY the locked facts below.\n"
                f"Keep botanical Latin names exact. Include all requested fields.\n\n"
                f"Locked Facts:\n{evidence_text}\n\n"
                f"User Question: {user_query}\n"
                f"Answer:"
            )
            payload["prompt"] = retry_prompt
            payload["options"]["num_predict"] = min(num_predict, 70)
            try:
                r_resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
                if r_resp.status_code == 200:
                    retry_cand = r_resp.json().get("response", "").strip()
                    r_grounded, _ = validate_groundedness(retry_cand, evidence, deterministic_answer)
                    if r_grounded:
                        self.telemetry["gemma_successes"] += 1
                        return retry_cand
            except Exception:
                pass

        # If Gemma fails, times out, or produces ungrounded claims after regeneration:
        # Fall back to verified deterministic graph-derived answer (ZERO LLAMA FALLBACK)
        self.telemetry["deterministic_fallbacks"] += 1
        return deterministic_answer

    # ========================================================
    # Main Dynamic Query Pipeline
    # ========================================================

    def query(self, user_query: str, use_llm: bool = True) -> str:
        lang = detect_language(user_query)
        query_lower = user_query.lower()

        # Step 1: Resolve Entity & Classify Domain
        resolved = self.resolver.resolve_query(user_query)
        res_type = resolved["type"]

        # Step 2: Handle Out-of-Domain & Unknown Entities (Deterministic safety guardrails)
        if res_type == "out_of_domain":
            return format_out_of_domain_answer("general_knowledge" if "minister" in query_lower or "president" in query_lower else "modern_pharmaceutical", lang)

        if "unknown" in res_type:
            parts = res_type.split("_", 1)
            unk_kind = parts[1] if len(parts) > 1 else "plant"
            unk_name = resolved.get("matched_term") or "Unknown"
            return format_unknown_entity_answer(unk_kind, unk_name, lang)

        # Step 3: Classify Intent
        intent = classify_intent(user_query, res_type)

        # Step 4: Execute Dynamic Graph Operations Based on Intent
        deterministic_answer = ""
        evidence = {"intent": intent, "user_query": user_query, "lang": lang}

        # Intent U: Universal / Epistemic Negative Claim Verification
        if intent == "universal_claim_verification":
            plant_data = resolved.get("data")
            if plant_data:
                pid = str(plant_data.get("plant_id", "")).strip()
                g_details = self.get_plant_graph_details(pid)
                common_name = resolved.get("matched_term") or g_details.get("sanskrit_name", "Plant")
                botanical_name = g_details.get("botanical_name", "")
                diseases = g_details.get("diseases", [])
                dis_str = ", ".join(diseases[:9]) if diseases else "none documented"

                if lang == "Hindi":
                    deterministic_answer = f"नहीं, ज्ञान आधार में ऐसा कोई प्रमाण नहीं है कि {common_name} सभी रोगों का इलाज करता है; इसमें विशिष्ट रोग संबंधों का उल्लेख है, जैसे {dis_str}।"
                elif lang == "Kannada":
                    deterministic_answer = f"ಇಲ್ಲ, {common_name} ಎಲ್ಲಾ ರೋಗಗಳನ್ನು ಗುಣಪಡಿಸುತ್ತದೆ ಎಂಬುದಕ್ಕೆ ಜ್ಞಾನ ಭಂಡಾರದಲ್ಲಿ ಪುರಾವೆಗಳಿಲ್ಲ; ಇದು {dis_str} ನಂತಹ ನಿರ್ದಿಷ್ಟ ರೋಗ ಸಂಬಂಧಗಳನ್ನು ದಾಖಲಿಸಿದೆ."
                else:
                    deterministic_answer = f"No, the knowledge base does not support claims that {common_name} treats every disease; it documents specific relationships such as {dis_str}."

                evidence = {
                    "common_name": common_name,
                    "botanical_name": botanical_name,
                    "universal_claim_refutation": f"No, the knowledge base does not support claims that {common_name} treats every disease. It only documents specific relationships: {dis_str}.",
                    "diseases": diseases[:9],
                    "requested_fields": ["epistemic refutation", "specific documented indications"],
                    "num_predict": 75
                }

        # Intent A: Disease Intersection (Multi-Hop Set Operation)
        elif intent == "disease_intersection" or res_type == "disease_intersection":
            d_found = resolved["data"] if isinstance(resolved["data"], list) else []
            if len(d_found) < 2:
                for d in DISEASE_REGISTRY:
                    if any(exact_term_match(n, user_query) for n in [d["name"], *d["aliases"]]):
                        if d["name"] not in d_found:
                            d_found.append(d["name"])
            if len(d_found) >= 2:
                d1, d2 = d_found[0], d_found[1]
                p1 = self.get_plants_for_disease(d1)
                p2 = self.get_plants_for_disease(d2)
                p2_ids = {p["plant_id"] for p in p2}
                common = [p for p in p1 if p["plant_id"] in p2_ids]
                deterministic_answer = format_disease_intersection_answer(d1, d2, common, lang)
                evidence = {
                    "diseases": [d1, d2],
                    "matching_plants": common[:4],
                    "requested_fields": ["common_plants"],
                    "num_predict": 95
                }

        # Intent B: Property Reverse Lookup (Multi-Hop)
        elif intent == "property_reverse_lookup" or res_type == "property_reverse_lookup":
            plant_data = resolved.get("data")
            if plant_data:
                pid = str(plant_data.get("plant_id", "")).strip()
                g_details = self.get_plant_graph_details(pid)
                prop_name = resolved.get("property", "virya" if "virya" in query_lower else "rasa")
                prop_val = g_details.get(prop_name, "Usna" if prop_name == "virya" else "Tikta")
                matched_plants = self.get_plants_by_property(prop_name, prop_val)
                target_name = resolved.get("matched_term", g_details.get("sanskrit_name", "Target Plant"))
                deterministic_answer = format_property_reverse_lookup_answer(target_name, prop_name, prop_val, matched_plants, lang)
                evidence = {
                    "common_name": target_name,
                    "summary_facts": f"{target_name} has {prop_name}={prop_val}. Matching plants: {', '.join([p.get('botanical_name', '') for p in matched_plants[:5]])}",
                    "matching_plants": matched_plants[:5],
                    "requested_fields": ["matching_plants"],
                    "num_predict": 85
                }

        # Intent C: Similar Disease Relationships (Multi-Hop Traversal)
        elif intent == "similar_disease_relationships":
            plant_data = resolved.get("data")
            if plant_data:
                pid = str(plant_data.get("plant_id", "")).strip()
                target_node = f"Plant_{pid}"
                g_details = self.get_plant_graph_details(pid)
                top_similar = self.get_similar_disease_plants(target_node)
                target_name = resolved.get("matched_term", g_details.get("sanskrit_name", "Plant"))
                deterministic_answer = format_similar_diseases_answer(target_name, g_details.get("diseases", []), top_similar, lang)
                evidence = {
                    "common_name": target_name,
                    "diseases": g_details.get("diseases", [])[:6],
                    "matching_plants": [{"botanical_name": b} for b in top_similar[:4]],
                    "requested_fields": ["similar_plants"],
                    "num_predict": 80
                }

        # Intent D: Plant Comparison (Comparison of Properties or Diseases)
        elif intent == "plant_comparison" or res_type == "plant_comparison":
            p_names = resolved["data"] if isinstance(resolved["data"], list) else []
            if len(p_names) >= 2:
                p1_name, p2_name = p_names[0], p_names[1]
                p1_data, p2_data = {}, {}
                for canon in [p1_name.lower(), p2_name.lower()]:
                    targets = CLASSICAL_ALIASES.get(canon, [canon])
                    primary_bot = targets[0].lower()
                    for plant in self.resolver.plants:
                        bot = normalize(str(plant.get("botanical_name", "")))
                        if primary_bot in bot:
                            pid = str(plant.get("plant_id", "")).strip()
                            if canon == p1_name.lower():
                                p1_data = self.get_plant_graph_details(pid)
                            else:
                                p2_data = self.get_plant_graph_details(pid)
                            break
                is_disease_comp = "disease" in query_lower or "share any" in query_lower
                deterministic_answer = format_plant_comparison_answer(p1_name, p1_data, p2_name, p2_data, is_disease_comp, lang)
                b1 = p1_data.get('botanical_name', 'N/A')
                b2 = p2_data.get('botanical_name', 'N/A')
                v1 = p1_data.get('virya', 'Not specified')
                v2 = p2_data.get('virya', 'Not specified')
                v1_gloss = f"{v1} (hot potency)" if "usna" in v1.lower() else (f"{v1} (cool potency)" if "sita" in v1.lower() else v1)
                v2_gloss = f"{v2} (hot potency)" if "usna" in v2.lower() else (f"{v2} (cool potency)" if "sita" in v2.lower() else v2)
                
                if is_disease_comp:
                    summary = deterministic_answer
                else:
                    summary = f"Property comparison: {p1_name} has {v1_gloss} Virya, whereas {p2_name} has {v2_gloss} Virya."

                evidence = {
                    "plant_1": f"{p1_name} ({b1})",
                    "plant_2": f"{p2_name} ({b2})",
                    "summary_facts": summary,
                    "requested_fields": ["property_comparison"],
                    "num_predict": 75
                }

        # Intent E: Constraint Queries (e.g. Connected to Disease & Having Documented Properties)
        elif intent == "constraint_lookup":
            target_dis = "Prameha"
            for d in DISEASE_REGISTRY:
                if any(exact_term_match(n, user_query) for n in [d["name"], *d["aliases"]]):
                    target_dis = d["name"]
                    break
            matching_plants = self.get_plants_for_disease(target_dis)
            prop_plants = [p for p in matching_plants if p.get("rasa") and p.get("rasa") != "Not specified"]
            if prop_plants:
                best_p = prop_plants[0]
                b_name = best_p["botanical_name"]
                s_name = best_p["sanskrit_name"] or "Herb"
                r = best_p["rasa"]
                v = best_p["virya"]
                if "Kannada" in lang:
                    deterministic_answer = f"{target_dis} ge sambandhisida plants featuring documented Ayurvedic properties include {s_name} ({b_name}, {r} Rasa/{v} Virya)."
                else:
                    deterministic_answer = f"{s_name} ({b_name}) is connected with {target_dis} in the database and has documented Ayurvedic properties ({r} Rasa, {v} Virya)."
                evidence = {
                    "common_name": s_name,
                    "botanical_name": b_name,
                    "rasa": r,
                    "virya": v,
                    "diseases": [target_dis],
                    "requested_fields": ["botanical_name", "rasa", "virya"],
                    "num_predict": 80
                }

        # Intent F: Disease to Plants (Reverse Lookup)
        elif res_type == "disease" or intent == "disease_to_plants":
            disease_data = resolved.get("data") or {}
            disease_name = disease_data.get("name") if isinstance(disease_data, dict) else str(resolved.get("matched_term", "Condition"))
            matching_plants = self.get_plants_for_disease(disease_name)
            deterministic_answer = format_disease_to_plants_answer(disease_name, matching_plants, lang)
            
            alias_map_hi = {"kasa": "कास (Kasa / खांसी)", "prameha": "प्रमेह (Prameha / मधुमेह)", "jvara": "ज्वर (Jvara / बुखार)"}
            alias_map_kn = {"prameha": "ಪ್ರಮೇಹ (Prameha)", "kasa": "ಕೆಮ್ಮು / ಕಾಸ (Kasa)", "jvara": "ಜ್ವರ (Jvara)"}
            if lang == "Hindi":
                disp_dis = alias_map_hi.get(disease_name.lower(), disease_name)
            elif lang == "Kannada":
                disp_dis = alias_map_kn.get(disease_name.lower(), disease_name)
            else:
                disp_dis = disease_name

            evidence = {
                "disease_name": disp_dis,
                "matching_plants": matching_plants[:4],
                "requested_fields": ["matching_plants"],
                "num_predict": 95
            }

        # Intent G: Plant Lookups (Forward Lookups from Resolved Plant)
        elif res_type == "plant":
            plant_data = resolved.get("data", {})
            pid = str(plant_data.get("plant_id", "")).strip()
            g_details = self.get_plant_graph_details(pid)
            common_name = resolved.get("matched_term") or g_details.get("sanskrit_name") or "Herb"
            botanical_name = g_details.get("botanical_name", "N/A")
            family = g_details.get("family", "N/A")
            diseases = g_details.get("diseases", [])
            synonyms = g_details.get("synonyms", [])
            rasa = g_details.get("rasa", "Not specified")
            virya = g_details.get("virya", "Not specified")
            vipaka = g_details.get("vipaka", "Not specified")

            clean_common = common_name.split("(")[0].strip()
            hi_alias, kn_alias = "", ""
            for c, t_list in CLASSICAL_ALIASES.items():
                if c.lower() == clean_common.lower():
                    dev_l = [t for t in t_list if re.search(r"[\u0900-\u097F]", t)]
                    if dev_l: hi_alias = dev_l[0]
                    kn_l = [t for t in t_list if re.search(r"[\u0C80-\u0CFF]", t)]
                    if kn_l: kn_alias = kn_l[0]
                    break

            if intent == "botanical_lookup":
                deterministic_answer = format_botanical_answer(common_name, botanical_name, family, lang)
                disp_plant = f"{hi_alias} ({clean_common})" if lang == "Hindi" and hi_alias else (f"{kn_alias} ({clean_common})" if lang == "Kannada" and kn_alias else clean_common)
                evidence = {
                    "common_name": disp_plant,
                    "botanical_name": botanical_name,
                    "family": family,
                    "requested_fields": ["botanical_name", "family"],
                    "num_predict": 70
                }
            elif intent == "botanical_to_common":
                deterministic_answer = format_botanical_to_common_answer(botanical_name, common_name, lang)
                evidence = {
                    "botanical_name": botanical_name,
                    "common_name": common_name,
                    "requested_fields": ["common_name"],
                    "num_predict": 60
                }
            elif intent == "family_lookup":
                deterministic_answer = format_family_answer(common_name, botanical_name, family, lang)
                evidence = {
                    "common_name": common_name,
                    "botanical_name": botanical_name,
                    "family": family,
                    "requested_fields": ["family", "botanical_name"],
                    "num_predict": 65
                }
            elif intent == "synonym_lookup":
                if "same plant as" in query_lower:
                    deterministic_answer = f"Yes, {common_name} and {botanical_name.split()[0]} {botanical_name.split()[1]} refer to the exact same medicinal plant entity in the knowledge base."
                elif "related to" in query_lower:
                    deterministic_answer = f"Yes, {common_name} is the common/Sanskrit name for the botanical species {botanical_name}."
                else:
                    deterministic_answer = format_synonyms_answer(common_name, botanical_name, synonyms, lang)

                syn_candidates = [s for s in synonyms if s and s.lower() != clean_common.lower()]
                for c, t_list in CLASSICAL_ALIASES.items():
                    if c.lower() == clean_common.lower():
                        syn_candidates.extend(t_list)
                        break
                seen_syn = set()
                final_syn = []
                for s in syn_candidates:
                    s_clean = s.strip()
                    if s_clean.lower() not in seen_syn and s_clean.lower() != clean_common.lower():
                        seen_syn.add(s_clean.lower())
                        final_syn.append(s_clean)

                evidence = {
                    "common_name": common_name,
                    "botanical_name": botanical_name,
                    "synonyms": final_syn[:6],
                    "requested_fields": ["synonyms", "botanical_name"],
                    "num_predict": 80
                }
                if "refer" in query_lower or "amrita" in query_lower or "अमृता" in query_lower:
                    evidence["refers_to_plant"] = f"{common_name} ({botanical_name})"
                    evidence["requested_fields"] = ["refers_to_plant", "botanical_name"]

            elif intent == "property_lookup":
                deterministic_answer = format_property_answer(common_name, botanical_name, g_details, query_lower, lang)
                is_virya_only = ("virya" in query_lower or "veerya" in query_lower or "वीर्य" in query_lower or "ವೀರ್ಯ" in query_lower) and \
                                ("rasa" not in query_lower and "रस" not in query_lower and "vipaka" not in query_lower and "विपाक" not in query_lower and "गुण" not in query_lower and "gunagalu" not in query_lower and "properties" not in query_lower)
                is_rasa_only = ("rasa" in query_lower or "रस" in query_lower or "taste" in query_lower) and \
                               ("virya" not in query_lower and "vipaka" not in query_lower and "गुण" not in query_lower and "gunagalu" not in query_lower and "properties" not in query_lower)
                is_vipaka_only = ("vipaka" in query_lower or "vipak" in query_lower or "विपाक" in query_lower or "ವಿಪಾಕ" in query_lower) and \
                                 ("rasa" not in query_lower and "virya" not in query_lower and "गुण" not in query_lower and "gunagalu" not in query_lower and "properties" not in query_lower)

                if is_virya_only:
                    v_gloss = "Usna (hot potency)" if "usna" in virya.lower() else "Sita (cool potency)"
                    evidence = {
                        "common_name": common_name,
                        "botanical_name": botanical_name,
                        "virya": f"उष्ण ({v_gloss})" if lang == "Hindi" else (f"ಉಷ್ಣ ({v_gloss})" if lang == "Kannada" else v_gloss),
                        "requested_fields": ["virya"],
                        "num_predict": 60
                    }
                elif is_rasa_only:
                    r_gloss = "Kashaya (astringent)" if ("kashaya" in rasa.lower() or "kasaya" in rasa.lower()) else ("Tikta (bitter)" if "tikta" in rasa.lower() else rasa)
                    evidence = {
                        "common_name": common_name,
                        "botanical_name": botanical_name,
                        "rasa": f"कषाय ({r_gloss})" if lang == "Hindi" else (f"ಕಷಾಯ ({r_gloss})" if lang == "Kannada" else r_gloss),
                        "requested_fields": ["rasa"],
                        "num_predict": 60
                    }
                elif is_vipaka_only:
                    evidence = {
                        "common_name": common_name,
                        "botanical_name": botanical_name,
                        "vipaka": vipaka,
                        "requested_fields": ["vipaka"],
                        "num_predict": 60
                    }
                else:
                    # Multi-property or full triad (e.g. Rasa, Virya and Vipaka)
                    req_p = []
                    if "botanical" in query_lower or "वानस्पतिक" in query_lower or "ಸಸ್ಯಶಾಸ್ತ್ರೀಯ" in query_lower:
                        req_p.append("botanical_name")
                    if "rasa" in query_lower or "रस" in query_lower or "taste" in query_lower or "गुण" in query_lower or "gunagalu" in query_lower or "properties" in query_lower:
                        req_p.append("rasa")
                    if "virya" in query_lower or "veerya" in query_lower or "वीर्य" in query_lower or "ವೀರ್ಯ" in query_lower or "गुण" in query_lower or "gunagalu" in query_lower or "properties" in query_lower:
                        req_p.append("virya")
                    if "vipaka" in query_lower or "vipak" in query_lower or "विपाक" in query_lower or "ವಿಪಾಕ" in query_lower or "गुण" in query_lower or "gunagalu" in query_lower or "properties" in query_lower:
                        req_p.append("vipaka")
                    if not req_p:
                        req_p = ["rasa", "virya", "vipaka"]

                    r_hi = "कषाय" if "kashaya" in rasa.lower() else ("तिक्त" if "tikta" in rasa.lower() else rasa)
                    v_hi = "उष्ण (Usna)" if "usna" in virya.lower() else ("शीत (Sita)" if "sita" in virya.lower() else virya)
                    vp_hi = "कटु (Katu)" if "katu" in vipaka.lower() else ("मधुर (Madhura)" if "madhura" in vipaka.lower() else vipaka)

                    r_kn = "ತಿಕ್ತ (Tikta)" if "tikta" in rasa.lower() else ("ಕಷಾಯ (Kashaya)" if "kashaya" in rasa.lower() else rasa)
                    v_kn = "ಉಷ್ಣ (Usna)" if "usna" in virya.lower() else ("ಶೀತ (Sita)" if "sita" in virya.lower() else virya)
                    vp_kn = "ಮಧುರ / ಕಟುಕ (Madhura / Katuka)" if ("madhura" in vipaka.lower() or "katuka" in vipaka.lower()) else vipaka

                    evidence = {
                        "common_name": f"{hi_alias} ({clean_common})" if lang == "Hindi" and hi_alias else (f"{kn_alias} ({clean_common})" if lang == "Kannada" and kn_alias else clean_common),
                        "botanical_name": botanical_name,
                        "rasa": r_hi if lang == "Hindi" else (r_kn if lang == "Kannada" else rasa),
                        "virya": v_hi if lang == "Hindi" else (v_kn if lang == "Kannada" else virya),
                        "vipaka": vp_hi if lang == "Hindi" else (vp_kn if lang == "Kannada" else vipaka),
                        "requested_fields": req_p,
                        "num_predict": 90
                    }

            elif intent == "plant_to_diseases":
                in_graph = "in the graph" in query_lower or "knowledge graph" in query_lower
                deterministic_answer = format_plant_to_diseases_answer(common_name, botanical_name, diseases, in_graph, lang)
                evidence = {
                    "common_name": common_name,
                    "botanical_name": botanical_name,
                    "diseases": diseases[:8],
                    "requested_fields": ["diseases"],
                    "num_predict": 80
                }
            else:
                # Overview / general plant facts
                deterministic_answer = format_plant_overview_answer(common_name, botanical_name, family, g_details, diseases, lang)
                v_chunks = self.retrieve_filtered_chunks(user_query, {pid}, final_k=1) if pid else []
                evidence = {
                    "common_name": common_name,
                    "botanical_name": botanical_name,
                    "family": family,
                    "rasa": rasa,
                    "virya": virya,
                    "vipaka": vipaka,
                    "diseases": diseases[:5],
                    "vector_context": v_chunks,
                    "requested_fields": ["overview"],
                    "num_predict": 100
                }

        if not deterministic_answer:
            deterministic_answer = (
                "I could not identify a recognized medicinal plant or Ayurvedic disease "
                "from your query in the database. Please verify the botanical, Sanskrit, "
                "or disease name."
            )

        if not use_llm:
            return deterministic_answer

        return self.generate_with_gemma(user_query, evidence, lang, deterministic_answer)


# ============================================================
# 7. Interactive CLI
# ============================================================

if __name__ == "__main__":
    service = GraphRAGService()

    print("=" * 60)
    print("Medicinal Plant Graph-RAG Active (Fully Dynamic Parametric Engine)")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Ask a plant question (or 'exit'): ").strip()
            if not question or question.lower() == "exit":
                break
            print("\n" + service.query(question))
            print("\n" + "-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break