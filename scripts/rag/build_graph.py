import json
import re
from pathlib import Path
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "integrated" / "medicinal_knowledge_base.json"
OUTPUT_GRAPH = PROJECT_ROOT / "data" / "knowledge_graph.gml"

# Ayurvedic property extraction patterns
PROPERTY_PATTERNS = {
    "rasa": r"(?:rasa|रस)\s*[:\-–]\s*([^.;\n]+)",
    "virya": r"(?:virya|veerya|वीर्य)\s*[:\-–]\s*([^.;\n]+)",
    "vipaka": r"(?:vipaka|paka|विपाक)\s*[:\-–]\s*([^.;\n]+)"
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
    {"name": "Vrana", "aliases": ["vrana", "ulcer", "wounds"]}
]


def clean_str(val):
    if not val or val == "Not available":
        return ""
    return str(val).strip()


def extract_dravyaguna(text):
    properties = {"rasa": None, "virya": None, "vipaka": None}
    
    rasa_words = {'tikta', 'katuka', 'katu', 'kashaya', 'madhura', 'amla', 'lavana'}
    virya_words = {'usna', 'ushna', 'sheeta', 'sita', 'sitala', 'sisira', 'hima'}
    vipaka_words = {'katu', 'katuka', 'madhura', 'amla', 'svadu'}

    # 1. Rasa
    m_rasa = re.search(r'([a-zA-Z\s,]+?)\s+in\s+rasa\b', text, re.IGNORECASE)
    if m_rasa:
        words = [w.strip() for w in m_rasa.group(1).replace(',', ' ').split() if w.strip()]
        found = [w for w in words if w.lower() in rasa_words]
        properties["rasa"] = ", ".join(dict.fromkeys(found)).title() if found else " ".join(words[-2:]).title()
    else:
        m = re.search(r'(?:rasa|रस)\s*[:\-–]\s*([^.;\n]+)', text, re.IGNORECASE)
        if m:
            properties["rasa"] = re.sub(r'[\u0900-\u097F|()]+', '', m.group(1)).strip().title()[:100]

    # 2. Virya
    m_virya = re.search(r'([a-zA-Z\s,]+?)\s+in\s+(?:veerya|virya|potency)\b', text, re.IGNORECASE)
    if m_virya:
        words = [w.strip() for w in m_virya.group(1).replace(',', ' ').split() if w.strip()]
        found = [w for w in words if w.lower() in virya_words]
        properties["virya"] = ", ".join(dict.fromkeys(found)).title() if found else words[-1].title()
    else:
        m = re.search(r'(?:virya|veerya|वीर्य)\s*[:\-–]\s*([^.;\n]+)', text, re.IGNORECASE)
        if m:
            properties["virya"] = re.sub(r'[\u0900-\u097F|()]+', '', m.group(1)).strip().title()[:100]

    # 3. Vipaka
    m_vipaka = re.search(r'([a-zA-Z\s,]+?)\s+in\s+(?:paka|vipaka)\b', text, re.IGNORECASE)
    if m_vipaka:
        words = [w.strip() for w in m_vipaka.group(1).replace(',', ' ').split() if w.strip()]
        found = [w for w in words if w.lower() in vipaka_words]
        properties["vipaka"] = ", ".join(dict.fromkeys(found)).title() if found else words[-1].title()
    else:
        m = re.search(r'(?:vipaka|paka|विपाक)\s*[:\-–]\s*([^.;\n]+)', text, re.IGNORECASE)
        if m:
            properties["vipaka"] = re.sub(r'[\u0900-\u097F|()]+', '', m.group(1)).strip().title()[:100]

    return properties


def extract_diseases(full_text):
    matched = set()
    norm_text = full_text.lower()
    for d in DISEASE_DEFINITIONS:
        for alias in d["aliases"]:
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            if re.search(pattern, norm_text):
                matched.add(d["name"])
                break
    return list(matched)


def build_graph():
    print(f"Loading knowledge base: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.MultiDiGraph()
    print(f"Building Knowledge Graph from {len(data)} plants...")

    for plant in data:
        plant_id = str(plant.get("plant_id", "")).strip()
        botanical_name = clean_str(plant.get("botanical_name", "Unknown"))
        family = clean_str(plant.get("family", ""))
        vernacular = clean_str(plant.get("vernacular_names", ""))

        # Collect full textual context for property & disease extraction
        text_corpus = ""
        for s in plant.get("shlokas", []):
            if isinstance(s, dict):
                text_corpus += " " + " ".join([str(v) for v in s.values()])
            elif isinstance(s, str):
                text_corpus += " " + s
        for ident in plant.get("ayurvedic_identity", []):
            if isinstance(ident, dict):
                text_corpus += " " + " ".join([str(v) for v in ident.values()])

        # Extract Dravyaguna
        dravyaguna = extract_dravyaguna(text_corpus)
        sanskrit_name = ""
        if plant.get("ayurvedic_identity"):
            for ident in plant["ayurvedic_identity"]:
                if isinstance(ident, dict) and ident.get("drug_name"):
                    sanskrit_name = clean_str(ident.get("drug_name"))
                    break

        plant_node_id = f"Plant_{plant_id}"

        # 1. Add Plant Node with structured properties
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
            vernacular_names=vernacular[:400]
        )

        # 2. Plant -> Family
        if family:
            fam_node_id = f"Family_{family.lower()}"
            G.add_node(fam_node_id, label="Family", name=family)
            G.add_edge(plant_node_id, fam_node_id, relation="BELONGS_TO")

        # 3. Plant -> System
        for sys_name in plant.get("systems", []):
            if clean_str(sys_name):
                sys_node_id = f"System_{clean_str(sys_name).lower()}"
                G.add_node(sys_node_id, label="System", name=clean_str(sys_name))
                G.add_edge(plant_node_id, sys_node_id, relation="USED_IN")

        # 4. Plant -[:TREATS]-> Disease (Single direction, backwards-traversable)
        diseases = extract_diseases(text_corpus)
        for dis_name in diseases:
            dis_node_id = f"Disease_{dis_name.lower()}"
            G.add_node(dis_node_id, label="Disease", name=dis_name)
            G.add_edge(plant_node_id, dis_node_id, relation="TREATS")

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