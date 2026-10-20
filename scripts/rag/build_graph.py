
import hashlib
import json
import re
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "integrated" / "medicinal_knowledge_base.json"
OUTPUT_GRAPH = PROJECT_ROOT / "data" / "knowledge_graph.gml"

PROPERTY_PATTERNS = {
    "rasa": r"(?:rasa|रस)\s*[:\-–]\s*([^.;\n]+)",
    "virya": r"(?:virya|veerya|वीर्य)\s*[:\-–]\s*([^.;\n]+)",
    "vipaka": r"(?:vipaka|paka|विपाक)\s*[:\-–]\s*([^.;\n]+)",
}

DISEASE_DEFINITIONS = [
    {"name": "Prameha", "aliases": ["prameha", "meha", "diabetes", "urinary disorder"]},
    {"name": "Jvara", "aliases": ["jvara", "fever"]},
    {"name": "Arsa", "aliases": ["arsa", "arsha", "piles", "hemorrhoids"]},
    {"name": "Kustha", "aliases": ["kustha", "kushtha", "skin disease", "skin diseases", "eczema"]},
    {"name": "Krmi", "aliases": ["krmi", "krimi", "worms", "parasites"]},
    {"name": "Kasa", "aliases": ["kasa", "cough", "bronchitis"]},
    {"name": "Svasa", "aliases": ["svasa", "shvasa", "asthma", "dyspnoea"]},
    {"name": "Pandu", "aliases": ["pandu", "anemia"]},
    {"name": "Grahani", "aliases": ["grahani", "ibs", "digestive disorder", "sprue"]},
    {"name": "Kamala", "aliases": ["kamala", "jaundice", "liver disorder"]},
    {"name": "Sopha", "aliases": ["sopha", "shotha", "inflammation", "swelling"]},
    {"name": "Vrana", "aliases": ["vrana", "ulcer", "wounds"]},
]


def clean_str(val):
    if not val or val == "Not available":
        return ""
    return str(val).strip()


def normalize_shloka_text(text):
    """Normalize case and whitespace to identify exact OCR-text duplicates."""
    return " ".join(str(text).strip().lower().split())


def extract_dravyaguna(text):
    properties = {"rasa": None, "virya": None, "vipaka": None}

    rasa_words = {
        "tikta", "katuka", "katu", "kashaya",
        "madhura", "amla", "lavana",
    }
    virya_words = {
        "usna", "ushna", "sheeta", "sita",
        "sitala", "sisira", "hima",
    }
    vipaka_words = {
        "katu", "katuka", "madhura", "amla", "svadu",
    }

    m_rasa = re.search(
        r"([a-zA-Z\s,]+?)\s+in\s+rasa\b",
        text,
        re.IGNORECASE,
    )
    if m_rasa:
        words = [
            w.strip()
            for w in m_rasa.group(1).replace(",", " ").split()
            if w.strip()
        ]
        found = [w for w in words if w.lower() in rasa_words]
        properties["rasa"] = (
            ", ".join(dict.fromkeys(found)).title()
            if found
            else " ".join(words[-2:]).title()
        )
    else:
        m = re.search(
            r"(?:rasa|रस)\s*[:\-–]\s*([^.;\n]+)",
            text,
            re.IGNORECASE,
        )
        if m:
            properties["rasa"] = re.sub(
                r"[\u0900-\u097F|()]+", "", m.group(1)
            ).strip().title()[:100]

    m_virya = re.search(
        r"([a-zA-Z\s,]+?)\s+in\s+(?:veerya|virya|potency)\b",
        text,
        re.IGNORECASE,
    )
    if m_virya:
        words = [
            w.strip()
            for w in m_virya.group(1).replace(",", " ").split()
            if w.strip()
        ]
        found = [w for w in words if w.lower() in virya_words]
        properties["virya"] = (
            ", ".join(dict.fromkeys(found)).title()
            if found
            else words[-1].title()
        )
    else:
        m = re.search(
            r"(?:virya|veerya|वीर्य)\s*[:\-–]\s*([^.;\n]+)",
            text,
            re.IGNORECASE,
        )
        if m:
            properties["virya"] = re.sub(
                r"[\u0900-\u097F|()]+", "", m.group(1)
            ).strip().title()[:100]

    m_vipaka = re.search(
        r"([a-zA-Z\s,]+?)\s+in\s+(?:paka|vipaka)\b",
        text,
        re.IGNORECASE,
    )
    if m_vipaka:
        words = [
            w.strip()
            for w in m_vipaka.group(1).replace(",", " ").split()
            if w.strip()
        ]
        found = [w for w in words if w.lower() in vipaka_words]
        properties["vipaka"] = (
            ", ".join(dict.fromkeys(found)).title()
            if found
            else words[-1].title()
        )
    else:
        m = re.search(
            r"(?:vipaka|paka|विपाक)\s*[:\-–]\s*([^.;\n]+)",
            text,
            re.IGNORECASE,
        )
        if m:
            properties["vipaka"] = re.sub(
                r"[\u0900-\u097F|()]+", "", m.group(1)
            ).strip().title()[:100]

    return properties


def extract_diseases(full_text):
    matched = set()
    norm_text = full_text.lower()

    for disease in DISEASE_DEFINITIONS:
        for alias in disease["aliases"]:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", norm_text):
                matched.add(disease["name"])
                break

    return list(matched)


def identity_status_for(plant, drug_name):
    """Return the status for this plant's exact drug-name identity, if present."""
    target = clean_str(drug_name).casefold()
    if not target:
        return "Unknown"

    statuses = []
    for ident in plant.get("ayurvedic_identity", []) or []:
        if not isinstance(ident, dict):
            continue
        if clean_str(ident.get("drug_name", "")).casefold() == target:
            status = clean_str(ident.get("status", ""))
            if status:
                statuses.append(status)

    # Prefer the explicit accepted status only when it is present for this exact name.
    for status in statuses:
        if status.casefold() == "accepted source":
            return status
    return statuses[0] if statuses else "Unknown"


def has_accepted_identity(plant, drug_name):
    """Return True only for a matching identity marked Accepted source."""
    return identity_status_for(plant, drug_name).casefold() == "accepted source"


def build_graph():
    print(f"Loading knowledge base: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect OCR texts attached to multiple botanical species.
    # Shared evidence is eligible only for a matching Accepted source identity.
    shloka_owners = {}

    for plant in data:
        botanical_name = clean_str(plant.get("botanical_name", ""))

        for shloka in plant.get("shlokas", []) or []:
            if not isinstance(shloka, dict):
                continue

            text = str(shloka.get("ocr_text", "")).strip()
            if len(text) < 40:
                continue

            key = hashlib.sha256(
                normalize_shloka_text(text).encode("utf-8")
            ).hexdigest()

            shloka_owners.setdefault(key, set()).add(botanical_name)

    shared_count = sum(
        1 for owners in shloka_owners.values() if len(owners) > 1
    )
    print(f"Shared shloka groups detected: {shared_count}")

    G = nx.MultiDiGraph()
    print(f"Building Knowledge Graph from {len(data)} plants...")

    for plant in data:
        plant_id = str(plant.get("plant_id", "")).strip()
        botanical_name = clean_str(plant.get("botanical_name", "Unknown"))
        family = clean_str(plant.get("family", ""))
        vernacular = clean_str(plant.get("vernacular_names", ""))

        # Keep evidence streams separate: species-specific facts are extracted
        # only from unique OCR or OCR explicitly accepted for this plant/drug.
        property_evidence = []
        treatment_evidence = []
        shared_mention_evidence = []
        all_source_mentions = []

        for shloka in plant.get("shlokas", []) or []:
            if not isinstance(shloka, dict):
                continue

            text = str(shloka.get("ocr_text", "")).strip()
            if len(text) < 40:
                continue

            key = hashlib.sha256(
                normalize_shloka_text(text).encode("utf-8")
            ).hexdigest()
            owners = shloka_owners.get(key, set())
            shloka_drug = clean_str(shloka.get("drug_name", ""))
            status = identity_status_for(plant, shloka_drug)
            # Preserve every OCR disease mention as source evidence. This is
            # separate from the conservative TREATS candidate relation below.
            all_source_mentions.append((text, key, shloka_drug, status, len(owners) > 1))

            if len(owners) <= 1:
                property_evidence.append(text)
                treatment_evidence.append((text, key, shloka_drug, status, False))
            elif has_accepted_identity(plant, shloka_drug):
                property_evidence.append(text)
                treatment_evidence.append((text, key, shloka_drug, status, True))
            else:
                # Do not discard shared text: retain its disease mentions with
                # provenance, but do not convert them into plant-level TREATS claims.
                shared_mention_evidence.append((text, key, shloka_drug, status))

        property_corpus = " ".join(property_evidence)
        dravyaguna = extract_dravyaguna(property_corpus)

        sanskrit_name = ""
        for ident in plant.get("ayurvedic_identity", []) or []:
            if isinstance(ident, dict) and ident.get("drug_name"):
                sanskrit_name = clean_str(ident.get("drug_name"))
                break

        plant_node_id = f"Plant_{plant_id}"

        G.add_node(
            plant_node_id,
            label="Plant",
            plant_id=plant_id,
            botanical_name=botanical_name,
            sanskrit_name=sanskrit_name,
            family=family,
            rasa=dravyaguna["rasa"] or "Not specified",
            virya=dravyaguna["virya"] or "Not specified",
            vipaka=dravyaguna["vipaka"] or "Not specified",
            vernacular_names=vernacular[:400],
        )

        if family:
            family_node_id = f"Family_{family.lower()}"
            G.add_node(family_node_id, label="Family", name=family)
            G.add_edge(
                plant_node_id,
                family_node_id,
                relation="BELONGS_TO",
            )

        for system_name in plant.get("systems", []):
            if clean_str(system_name):
                system_node_id = (
                    f"System_{clean_str(system_name).lower()}"
                )
                G.add_node(
                    system_node_id,
                    label="System",
                    name=clean_str(system_name),
                )
                G.add_edge(
                    plant_node_id,
                    system_node_id,
                    relation="USED_IN",
                )

        # Unique evidence and shared evidence accepted for this exact drug
        # identity may create candidate TREATS edges. These remain keyword-based
        # candidates, not clinically validated claims.
        for text, evidence_hash, source_drug, identity_status, is_shared in treatment_evidence:
            for disease_name in extract_diseases(text):
                disease_node_id = f"Disease_{disease_name.lower()}"
                G.add_node(disease_node_id, label="Disease", name=disease_name)
                G.add_edge(
                    plant_node_id,
                    disease_node_id,
                    relation="TREATS",
                    evidence_hash=evidence_hash,
                    source_drug=source_drug or "Unknown",
                    identity_status=identity_status,
                    shared_evidence=bool(is_shared),
                    evidence_type="keyword_candidate",
                )

        # Preserve disease mentions from all OCR sources separately from treatment
        # candidates. A keyword mention is not, by itself, proof that this plant
        # treats the disease. The graph-RAG layer labels these as source mentions.
        for text, evidence_hash, source_drug, identity_status, is_shared in all_source_mentions:
            for disease_name in extract_diseases(text):
                disease_node_id = f"Disease_{disease_name.lower()}"
                G.add_node(disease_node_id, label="Disease", name=disease_name)
                G.add_edge(
                    plant_node_id,
                    disease_node_id,
                    relation="MENTIONS_DISEASE",
                    evidence_hash=evidence_hash,
                    source_drug=source_drug or "Unknown",
                    identity_status=identity_status,
                    shared_evidence=bool(is_shared),
                    evidence_type="ocr_source_mention_only",
                )

    OUTPUT_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(G, str(OUTPUT_GRAPH))

    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH REBUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    print(f"Saved to   : {OUTPUT_GRAPH}\n")


if __name__ == "__main__":
    build_graph()
