import json
import re
from pathlib import Path
from collections import defaultdict

# ==========================================================
# Base Directory
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# ==========================================================
# Input Files
# ==========================================================

MEDICINAL_FILE = BASE_DIR / "data" / "cleaned" / "medicinal_plants_clean.json"
SYSTEMS_FILE = BASE_DIR / "data" / "cleaned" / "systems_links_clean.json"
DISTRIBUTION_FILE = BASE_DIR / "data" / "cleaned" / "distribution_clean.json"
IDENTITY_FILE = BASE_DIR / "data" / "cleaned" / "ayurvedic_identity_clean.json"
SHLOKAS_FILE = BASE_DIR / "data" / "cleaned" / "shlokas_clean.json"

# ==========================================================
# Output File
# ==========================================================

OUTPUT_DIR = BASE_DIR / "data" / "integrated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "medicinal_knowledge_base.json"

# ==========================================================
# Helper Functions
# ==========================================================

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save JSON data."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def normalize_botanical_name(name):
    """
    Normalize botanical names for matching.
    Example:
        Azadirachta indica A. Juss.
    becomes:
        azadirachta indica
    """
    if not name:
        return ""

    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)

    parts = name.split()

    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"

    return name


# ==========================================================
# Load Cleaned Datasets
# ==========================================================

print("\nLoading cleaned datasets...\n")

medicinal = load_json(MEDICINAL_FILE)
systems = load_json(SYSTEMS_FILE)
distribution = load_json(DISTRIBUTION_FILE)
identity = load_json(IDENTITY_FILE)
shlokas = load_json(SHLOKAS_FILE)

print(f"Medicinal Plants : {len(medicinal)}")
print(f"Systems Links    : {len(systems)}")
print(f"Distribution     : {len(distribution)}")
print(f"Identity         : {len(identity)}")
print(f"Shlokas          : {len(shlokas)}")

# ==========================================================
# Create Knowledge Base
# ==========================================================

print("\nCreating knowledge base...")

knowledge_base = {}

for plant in medicinal:

    plant_id = plant["plant_id"]

    knowledge_base[plant_id] = {
        "plant_id": plant["plant_id"],
        "botanical_name": plant["botanical_name"],
        "family": plant["family"],
        "systems": plant["systems"],
        "synonyms": plant["synonyms"],
        "vernacular_names": plant["vernacular_names"],
        "detail_url": plant["detail_url"],

        # Data to be attached later
        "distribution": {},
        "system_links": [],
        "ayurvedic_identity": [],
        "shlokas": []
    }

print(f"Knowledge base created with {len(knowledge_base)} unique plants.")

# ==========================================================
# Build Lookup Indexes
# ==========================================================

print("\nBuilding lookup indexes...")

plant_id_index = knowledge_base

botanical_name_index = {}
duplicate_names = defaultdict(list)

for plant_id, plant in knowledge_base.items():

    name = plant["botanical_name"].strip().lower()

    botanical_name_index[name] = plant_id
    duplicate_names[name].append(plant_id)

duplicates = {
    name: ids
    for name, ids in duplicate_names.items()
    if len(ids) > 1
}

print(f"Plant ID Index       : {len(plant_id_index)}")
print(f"Botanical Name Index : {len(botanical_name_index)}")
print(f"Duplicate Names      : {len(duplicates)}")

# ==========================================================
# Attach System Links
# ==========================================================

print("\nAttaching system links...")

attached = 0

for record in systems:

    plant_id = record["plant_id"]

    if plant_id in knowledge_base:

        knowledge_base[plant_id]["system_links"].append({
            "system": record["system_source"],
            "detail_url": record["detail_url"]
        })

        attached += 1

print(f"System links attached : {attached}")

# ==========================================================
# Attach Distribution Data
# ==========================================================

print("\nAttaching distribution data...")

distribution_matched = 0
distribution_unmatched = 0

for record in distribution:

    distribution_name = normalize_botanical_name(record["plant_name"])

    found = False

    for plant in knowledge_base.values():

        medicinal_name = normalize_botanical_name(
            plant["botanical_name"]
        )

        if medicinal_name == distribution_name:

            plant["distribution"] = {
                "map_id": record["map_id"],
                "distribution_map_url": record["distribution_map_url"]
            }

            distribution_matched += 1
            found = True
            break

    if not found:
        distribution_unmatched += 1

print(f"Distribution matched   : {distribution_matched}")
print(f"Distribution unmatched : {distribution_unmatched}")

# ==========================================================
# Attach Ayurvedic Identity
# ==========================================================

print("\nAttaching Ayurvedic Identity...")

identity_matched = 0
identity_unmatched = 0

for record in identity:

    identity_name = normalize_botanical_name(
        record["botanical_correlation"]
    )

    found = False

    for plant in knowledge_base.values():

        medicinal_name = normalize_botanical_name(
            plant["botanical_name"]
        )

        if medicinal_name == identity_name:

            plant["ayurvedic_identity"].append({

                "drug_name": record["drug_name"],
                "sanskrit_name": record["sanskrit_name"],
                "transliteration": record["transliteration"],
                "ref_drug_name": record["ref_drug_name"],
                "status": record["status"],
                "discussion": record["discussion"],
                "reference": record["reference"]

            })

            identity_matched += 1
            found = True
            break

    if not found:
        identity_unmatched += 1

print(f"Identity matched   : {identity_matched}")
print(f"Identity unmatched : {identity_unmatched}")

# ==========================================================
# Attach Shlokas
# ==========================================================

print("\nAttaching Shlokas...")

# Build a lookup table for shlokas
shloka_lookup = {}

for shloka in shlokas:
    key = shloka["drug_name"].strip().lower()
    shloka_lookup[key] = shloka

shloka_matched = 0

for plant in knowledge_base.values():

    for identity_record in plant["ayurvedic_identity"]:

        drug_name = identity_record["drug_name"].strip().lower()

        if drug_name in shloka_lookup:

            identity_record["shloka"] = {
                "image_urls": shloka_lookup[drug_name]["image_urls"],
                "ocr_text": shloka_lookup[drug_name]["ocr_text"]
            }

            shloka_matched += 1

print(f"Shlokas attached : {shloka_matched}")

# ==========================================================
# Save Checkpoint
# ==========================================================



# ==========================================================
# Save Final Knowledge Base
# ==========================================================

save_json(
    list(knowledge_base.values()),
    OUTPUT_FILE
)

print("\n==========================================")
print("Integration Completed Successfully!")
print("==========================================")
print(f"Total Plants            : {len(knowledge_base)}")
print(f"Systems Links           : {len(systems)}")
print(f"Distribution Matched    : {distribution_matched}")
print(f"Distribution Unmatched  : {distribution_unmatched}")
print(f"Identity Matched        : {identity_matched}")
print(f"Identity Unmatched      : {identity_unmatched}")
print(f"Unique Shlokas       : {len(shlokas)}")
print(f"Shloka Attachments   : {shloka_matched}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

