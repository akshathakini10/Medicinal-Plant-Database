import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "integrated"
    / "medicinal_knowledge_base.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "rag_documents"
)

# Create output directory if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print("Loading integrated knowledge base...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

print(f"Total plants loaded: {len(knowledge_base)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_list(value):
    """Convert a list into readable text."""

    if not value:
        return "Not available"

    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)

    return str(value)


def format_value(value):
    """Convert different JSON values into readable text."""

    if value is None:
        return "Not available"

    if isinstance(value, list):
        if not value:
            return "Not available"

        return "\n".join(str(item) for item in value)

    if isinstance(value, dict):
        if not value:
            return "Not available"

        lines = []

        for key, val in value.items():
            lines.append(f"{key}: {val}")

        return "\n".join(lines)

    return str(value)


# ============================================================
# CREATE ONE DOCUMENT PER PLANT
# ============================================================

print("\nCreating RAG documents...")

created = 0
skipped = 0

for plant in knowledge_base:

    plant_id = plant.get("plant_id", "")

    botanical_name = plant.get(
        "botanical_name",
        "Unknown"
    )

    family = plant.get(
        "family",
        "Not available"
    )

    systems = plant.get(
        "systems",
        []
    )

    synonyms = plant.get(
        "synonyms",
        []
    )

    vernacular_names = plant.get(
        "vernacular_names",
        "Not available"
    )

    detail_url = plant.get(
        "detail_url",
        "Not available"
    )

    distribution = plant.get(
        "distribution",
        {}
    )

    system_links = plant.get(
        "system_links",
        []
    )

    ayurvedic_identity = plant.get(
        "ayurvedic_identity",
        []
    )

    shlokas = plant.get(
        "shlokas",
        []
    )

    # --------------------------------------------------------
    # Start document
    # --------------------------------------------------------

    document = []

    document.append("MEDICINAL PLANT")
    document.append("=" * 60)

    document.append(f"Plant ID: {plant_id}")
    document.append(f"Botanical Name: {botanical_name}")
    document.append(f"Family: {family}")

    document.append("\nSYSTEMS OF MEDICINE")
    document.append("-" * 60)
    document.append(format_list(systems))

    document.append("\nSYNONYMS")
    document.append("-" * 60)
    document.append(format_list(synonyms))

    document.append("\nVERNACULAR NAMES")
    document.append("-" * 60)
    document.append(format_value(vernacular_names))

    document.append("\nDETAIL URL")
    document.append("-" * 60)
    document.append(str(detail_url))

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    document.append("\nDISTRIBUTION")
    document.append("-" * 60)

    if distribution:

        for key, value in distribution.items():
            document.append(
                f"{key}: {value}"
            )

    else:
        document.append("Not available")

    # --------------------------------------------------------
    # Medicinal System Links
    # --------------------------------------------------------

    document.append("\nMEDICINAL SYSTEM LINKS")
    document.append("-" * 60)

    if system_links:

        for link in system_links:

            if isinstance(link, dict):

                system = link.get(
                    "system",
                    "Unknown"
                )

                url = link.get(
                    "detail_url",
                    "Not available"
                )

                document.append(
                    f"System: {system}"
                )

                document.append(
                    f"URL: {url}"
                )

            else:
                document.append(str(link))

    else:
        document.append("Not available")

    # --------------------------------------------------------
    # Ayurvedic Identity
    # --------------------------------------------------------

    document.append("\nAYURVEDIC IDENTITY")
    document.append("-" * 60)

    if ayurvedic_identity:

        for index, identity in enumerate(
            ayurvedic_identity,
            start=1
        ):

            document.append(
                f"\nIdentity Record {index}"
            )

            if isinstance(identity, dict):

                for key, value in identity.items():

                    document.append(
                        f"{key}: {format_value(value)}"
                    )

            else:
                document.append(
                    str(identity)
                )

    else:
        document.append("Not available")

    # --------------------------------------------------------
    # Sanskrit Shlokas
    # --------------------------------------------------------

    document.append("\nSANSKRIT SHLOKAS AND MEDICINAL USE INFORMATION")
    document.append("-" * 60)

    if shlokas:

        for index, shloka in enumerate(
            shlokas,
            start=1
        ):

            document.append(
                f"\nShloka Record {index}"
            )

            if isinstance(shloka, dict):

                for key, value in shloka.items():

                    document.append(
                        f"{key}: {format_value(value)}"
                    )

            else:

                document.append(
                    str(shloka)
                )

    else:
        document.append("Not available")

    # --------------------------------------------------------
    # Combine document
    # --------------------------------------------------------

    document_text = "\n".join(document)

    # --------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------

    safe_id = str(plant_id).strip()

    if not safe_id:
        skipped += 1
        continue

    output_file = OUTPUT_DIR / f"{safe_id}.txt"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(document_text)

    created += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DOCUMENT PREPARATION COMPLETED")
print("=" * 60)

print(f"Plants loaded   : {len(knowledge_base)}")
print(f"Documents created: {created}")
print(f"Documents skipped: {skipped}")

print("\nOutput directory:")
print(OUTPUT_DIR)