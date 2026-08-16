import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "medicinal_plants.index"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

SEARCH_K = 50


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)

print(
    f"Vectors in index: {index.ntotal}"
)


# ============================================================
# LOAD METADATA
# ============================================================

print("Loading metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(
    f"Metadata records: {len(metadata)}"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# REMOVE BOTANICAL AUTHOR
# ============================================================

def normalize_botanical_name(
    name
):

    name = normalize_text(
        name
    )

    if not name:
        return ""

    # Remove common botanical-author suffixes.
    # Example:
    # Aquilaria malaccensis LAM.
    # -> aquilaria malaccensis

    author_patterns = [
        r"\b[a-z]{2,}\.$",
        r"\b[a-z]{2,}\s+[a-z]{2,}\.$",
    ]

    for pattern in author_patterns:

        name = re.sub(
            pattern,
            "",
            name
        ).strip()

    return name


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "are",
    "the",
    "medicinal",
    "uses",
    "use",
    "of",
    "is",
    "tell",
    "me",
    "about",
    "for",
    "does",
    "do",
    "how",
    "which",
    "can",
    "give",
    "information",
    "on",
    "and",
    "in",
    "to",
    "a",
    "an",
    "please",
    "plant",
    "plants",
    "mentioned",
    "properties",
    "property",
    "benefits",
    "benefit",
    "medicinally"
}


# ============================================================
# EXTRACT PLANT NAME FROM QUERY
# ============================================================

def extract_plant_name(
    query
):

    normalized = normalize_text(
        query
    )

    words = normalized.split()

    remaining = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(
        remaining
    ).strip()


# ============================================================
# EXACT NAME MATCH
# ============================================================

def names_match(
    query_name,
    database_name
):

    query_name = normalize_text(
        query_name
    )

    database_name = normalize_text(
        database_name
    )

    if not query_name or not database_name:
        return False

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if query_name == database_name:
        return True

    # --------------------------------------------------------
    # Botanical names may have author information.
    #
    # Example:
    #
    # Query:
    # Plumbago zeylanica
    #
    # Database:
    # Plumbago zeylanica L.
    # --------------------------------------------------------

    if database_name.startswith(
        query_name + " "
    ):

        remaining = database_name[
            len(query_name):
        ].strip()

        # Remaining text should normally be botanical
        # author information.
        #
        # Accept only if it does not look like another
        # completely different plant name.

        if remaining:

            return True

    return False


# ============================================================
# FIND EXACT PLANT
# ============================================================

def find_exact_plant(
    query
):

    query_name = extract_plant_name(
        query
    )

    if not query_name:

        return None

    print(
        f"Searching exact plant name: {query_name}"
    )

    # ========================================================
    # PASS 1
    # BOTANICAL NAME
    # ========================================================

    for item in metadata:

        botanical_name = item.get(
            "botanical_name",
            ""
        )

        if names_match(
            query_name,
            botanical_name
        ):

            print(
                "Exact botanical-name match found."
            )

            return item

    # ========================================================
    # PASS 2
    # STRUCTURED DRUG NAME
    # ========================================================

    for item in metadata:

        drug_name = item.get(
            "drug_name",
            ""
        )

        if names_match(
            query_name,
            drug_name
        ):

            print(
                "Exact drug-name match found."
            )

            return item

    # ========================================================
    # PASS 3
    # TRANSLITERATION
    # ========================================================

    for item in metadata:

        transliteration = item.get(
            "transliteration",
            ""
        )

        if names_match(
            query_name,
            transliteration
        ):

            print(
                "Exact transliteration match found."
            )

            return item

    # ========================================================
    # PASS 4
    # SEARCH DRUG_NAME INSIDE CHUNK
    #
    # IMPORTANT:
    # Match the COMPLETE phrase.
    #
    # Do NOT match individual words.
    # ========================================================

    escaped_name = re.escape(
        query_name
    )

    for item in metadata:

        text = item.get(
            "text",
            ""
        )

        if not text:
            continue

        pattern = re.compile(
            rf"drug[_\s]?name\s*:\s*"
            rf"{escaped_name}"
            rf"(?:\s|$|,|;)",
            flags=re.IGNORECASE
        )

        if pattern.search(text):

            print(
                "Exact drug-name match found inside chunk."
            )

            return item

    # ========================================================
    # PASS 5
    # SEARCH BOTANICAL NAME INSIDE CHUNK
    #
    # COMPLETE NAME ONLY
    # ========================================================

    for item in metadata:

        text = item.get(
            "text",
            ""
        )

        if not text:
            continue

        pattern = re.compile(
            rf"Botanical\s+Name\s*:\s*"
            rf"{escaped_name}"
            rf"(?:\s|$|,|;|\.)",
            flags=re.IGNORECASE
        )

        if pattern.search(text):

            print(
                "Exact botanical-name match found inside chunk."
            )

            return item

    return None


# ============================================================
# FAISS FALLBACK
# ============================================================

def find_best_plant_semantic(
    query
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype(
            "float32"
        ),
        SEARCH_K
    )

    if len(
        indices[0]
    ) == 0:

        return None

    for idx in indices[0]:

        if (
            idx >= 0
            and
            idx < len(metadata)
        ):

            return metadata[idx]

    return None


# ============================================================
# FIND BEST PLANT
# ============================================================

def find_best_plant(
    query
):

    exact_plant = find_exact_plant(
        query
    )

    if exact_plant is not None:

        return exact_plant

    print(
        "No exact plant-name match found."
    )

    print(
        "Using FAISS semantic search..."
    )

    return find_best_plant_semantic(
        query
    )


# ============================================================
# GET ALL CHUNKS FOR PLANT
# ============================================================

def get_plant_chunks(
    plant_id
):

    chunks = []

    for item in metadata:

        if item.get(
            "plant_id"
        ) == plant_id:

            chunks.append(
                item
            )

    chunks.sort(
        key=lambda x: x.get(
            "chunk_index",
            0
        )
    )

    return chunks


# ============================================================
# FIND MEDICINAL CHUNKS
# ============================================================

def get_medicinal_chunks(
    chunks
):

    medicinal = []

    medicinal_started = False

    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        )

        upper = text.upper()

        # ----------------------------------------------------
        # Medicinal section begins
        # ----------------------------------------------------

        if (
            "SANSKRIT SHLOKAS" in upper
            or
            "MEDICINAL USE INFORMATION" in upper
            or
            "SHLOKA RECORD" in upper
        ):

            medicinal_started = True

        if (
            medicinal_started
            or
            "OCR_TEXT:" in upper
        ):

            medicinal.append(
                chunk
            )

    return medicinal


# ============================================================
# EXTRACT OCR TEXT
# ============================================================

def extract_ocr_text(
    text
):

    if not text:
        return ""

    match = re.search(
        r"ocr_text\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:

        return match.group(
            1
        )

    return text


# ============================================================
# CLEAN OCR
# ============================================================

def clean_ocr(
    text
):

    if not text:
        return ""

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove Sanskrit characters
    # --------------------------------------------------------

    text = re.sub(
        r"[\u0900-\u097F]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # OCR corrections
    # --------------------------------------------------------

    replacements = {

        "alleviaies":
            "alleviates",

        "alleviaie":
            "alleviates",

        "Due io":
            "Due to",

        "due io":
            "due to",

        "vaia":
            "vata",

        "Vaia":
            "Vata",

        "Ioha":
            "loha",

        "iase":
            "rasa",

        "tikia":
            "tikta",

        "ushna":
            "usna"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # --------------------------------------------------------
    # Remove OCR separators
    # --------------------------------------------------------

    text = re.sub(
        r"[|]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# REMOVE SOURCE LABELS
# ============================================================

def remove_source_labels(
    text
):

    if not text:
        return ""

    text = re.sub(
        r"\([A-Za-z]{2,10}\)",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT HARA PHRASES
# ============================================================

def extract_hara_phrases(
    text
):

    results = []

    if not text:
        return results

    # ========================================================
    # vata kapha hara
    # ========================================================

    if re.search(
        r"\bvata\s+kapha\s+hara\b",
        text,
        flags=re.IGNORECASE
    ):

        results.append(
            "alleviates vata"
        )

        results.append(
            "alleviates kapha"
        )

    # ========================================================
    # kapha vata hara
    # ========================================================

    if re.search(
        r"\bkapha\s+vata\s+hara\b",
        text,
        flags=re.IGNORECASE
    ):

        results.append(
            "alleviates kapha"
        )

        results.append(
            "alleviates vata"
        )

    # ========================================================
    # pitta hara
    # ========================================================

    if re.search(
        r"\bpitta\s+hara\b",
        text,
        flags=re.IGNORECASE
    ):

        results.append(
            "alleviates pitta"
        )

    # ========================================================
    # vata hara
    # ========================================================

    if re.search(
        r"\bvata\s+hara\b",
        text,
        flags=re.IGNORECASE
    ):

        results.append(
            "alleviates vata"
        )

    # ========================================================
    # kapha hara
    # ========================================================

    if re.search(
        r"\bkapha\s+hara\b",
        text,
        flags=re.IGNORECASE
    ):

        results.append(
            "alleviates kapha"
        )

    # ========================================================
    # kustha hara
    # ========================================================

    if re.search(
        r"\bkustha\s+hara\b",
        text,
        flags=re.IGNORECASE
    ):

        results.append(
            "alleviates kustha"
        )

    # ========================================================
    # sruta netraruja hara
    # ========================================================

    if re.search(
        r"\bsruta\s+netraruja\s+hara\b",
        text,
        flags=re.IGNORECASE
    ):

        results.append(
            "alleviates sruta netraruja"
        )

    return results


# ============================================================
# EXTRACT EXPLICIT THERAPEUTIC STATEMENTS
# ============================================================

def extract_explicit_statements(
    text
):

    results = []

    if not text:
        return results

    text = clean_ocr(
        text
    )

    text = remove_source_labels(
        text
    )

    # ========================================================
    # CURES
    # ========================================================

    cure_matches = re.finditer(
        r"\bcures?\s+(.+?)(?="
        r"\bKrsna\s+is\b|"
        r"\bKrishna\s+is\b|"
        r"\bAnother\s+variety\b|"
        r"\bSynonyms?\s+of\b|"
        r"\bAgaru\s+is\b|"
        r"\bCitraka\s+is\b|"
        r"\bAgasti\s+is\b|"
        r"\.|$)",
        text,
        flags=re.IGNORECASE
    )

    for match in cure_matches:

        value = match.group(
            1
        ).strip()

        value = value.strip(
            " ,:;-"
        )

        # Remove accidental OCR continuation
        value = re.sub(
            r"\bKrsna\s+is.*$",
            "",
            value,
            flags=re.IGNORECASE
        )

        value = re.sub(
            r"\bKrishna\s+is.*$",
            "",
            value,
            flags=re.IGNORECASE
        )

        value = value.strip(
            " ,:;-"
        )

        if value:

            results.append(
                "cures " + value
            )

    # ========================================================
    # ALLEVIATES
    # ========================================================

    alleviate_matches = re.finditer(
        r"\balleviates?\s+(.+?)(?="
        r"\bincreases?\b|"
        r"\bcures?\b|"
        r"\bpacifies?\b|"
        r"\bAnother\s+variety\b|"
        r"\b[A-Z][a-z]+\s+is\b|"
        r"\.|$)",
        text,
        flags=re.IGNORECASE
    )

    for match in alleviate_matches:

        value = match.group(
            1
        ).strip()

        value = value.strip(
            " ,:;-"
        )

        # ----------------------------------------------------
        # Split comma-separated effects
        # ----------------------------------------------------

        parts = re.split(
            r",",
            value
        )

        for part in parts:

            part = part.strip(
                " .,:;-"
            )

            if not part:
                continue

            if len(
                part.split()
            ) <= 6:

                results.append(
                    "alleviates " + part
                )

    # ========================================================
    # INCREASES
    # ========================================================

    increase_matches = re.finditer(
        r"\bincreases?\s+(.+?)(?="
        r"\balleviates?\b|"
        r"\bcures?\b|"
        r"\bpacifies?\b|"
        r"\bAnother\s+variety\b|"
        r"\b[A-Z][a-z]+\s+is\b|"
        r"\.|$)",
        text,
        flags=re.IGNORECASE
    )

    for match in increase_matches:

        value = match.group(
            1
        ).strip()

        value = value.strip(
            " ,:;-"
        )

        parts = re.split(
            r",",
            value
        )

        for part in parts:

            part = part.strip(
                " .,:;-"
            )

            if not part:
                continue

            if len(
                part.split()
            ) <= 6:

                results.append(
                    "increases " + part
                )

    # ========================================================
    # PACIFIES
    # ========================================================

    pacify_matches = re.finditer(
        r"\bpacifies?\s+(.+?)(?="
        r"\balleviates?\b|"
        r"\bcures?\b|"
        r"\bincreases?\b|"
        r"\bAnother\s+variety\b|"
        r"\b[A-Z][a-z]+\s+is\b|"
        r"\.|$)",
        text,
        flags=re.IGNORECASE
    )

    for match in pacify_matches:

        value = match.group(
            1
        ).strip()

        value = value.strip(
            " ,:;-"
        )

        parts = re.split(
            r",",
            value
        )

        for part in parts:

            part = part.strip(
                " .,:;-"
            )

            if (
                part
                and
                len(part.split()) <= 6
            ):

                results.append(
                    "pacifies " + part
                )

    return results


# ============================================================
# REMOVE NON-THERAPEUTIC RESULTS
# ============================================================

def remove_non_therapeutic(
    results
):

    final = []

    blocked_phrases = [

        "sneha",
        "mangalya",
        "rasa",
        "guna",
        "veerya",
        "paka",
        "tvacya",
        "laghu",
        "snigdha",
        "ruksa",
        "usna",
        "ushna",
        "katu",
        "tikta",

        "sinks in water",
        "sinks in water like",

        "krsna is",
        "krishna is",

        "gunadhika",

        "synonyms",

        "another variety"
    ]

    for result in results:

        result = result.strip()

        result = re.sub(
            r"\s+",
            " ",
            result
        )

        result = result.strip(
            " .,:;-"
        )

        if not result:
            continue

        lower = result.lower()

        if any(
            phrase in lower
            for phrase in blocked_phrases
        ):

            continue

        final.append(
            result
        )

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    unique = []

    seen = set()

    for result in final:

        key = normalize_text(
            result
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            result
        )

    return unique


# ============================================================
# EXTRACT THERAPEUTIC INFORMATION
# ============================================================

def extract_therapeutic_information(
    medicinal_chunks
):

    results = []

    for chunk in medicinal_chunks:

        raw_text = chunk.get(
            "text",
            ""
        )

        if not raw_text:
            continue

        text = extract_ocr_text(
            raw_text
        )

        text = clean_ocr(
            text
        )

        if not text:
            continue

        # ----------------------------------------------------
        # Extract traditional hara expressions
        # ----------------------------------------------------

        results.extend(
            extract_hara_phrases(
                text
            )
        )

        # ----------------------------------------------------
        # Extract explicit English therapeutic statements
        # ----------------------------------------------------

        results.extend(
            extract_explicit_statements(
                text
            )
        )

    results = remove_non_therapeutic(
        results
    )

    return results


# ============================================================
# FORMAT FINAL ANSWER
# ============================================================

def format_answer(
    plant,
    therapeutic_results
):

    botanical_name = plant.get(
        "botanical_name",
        "Not available"
    )

    family = plant.get(
        "family",
        "Not available"
    )

    lines = []

    lines.append(
        f"Botanical Name: {botanical_name}"
    )

    lines.append(
        f"Family: {family}"
    )

    lines.append("")

    lines.append(
        "Medicinal uses:"
    )

    if not therapeutic_results:

        lines.append(
            "- No explicit therapeutic information "
            "was found in the database."
        )

    else:

        for result in therapeutic_results:

            lines.append(
                f"- {result}"
            )

    return "\n".join(
        lines
    )


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(
    question
):

    print(
        "\nRetrieving relevant information..."
    )

    # ========================================================
    # STEP 1: FIND CORRECT PLANT
    # ========================================================

    plant = find_best_plant(
        question
    )

    if plant is None:

        print(
            "\nNo relevant plant was found."
        )

        return

    plant_id = plant.get(
        "plant_id"
    )

    botanical_name = plant.get(
        "botanical_name",
        "Not available"
    )

    family = plant.get(
        "family",
        "Not available"
    )

    # ========================================================
    # STEP 2: GET ALL PLANT CHUNKS
    # ========================================================

    plant_chunks = get_plant_chunks(
        plant_id
    )

    # ========================================================
    # STEP 3: GET MEDICINAL CHUNKS
    # ========================================================

    medicinal_chunks = get_medicinal_chunks(
        plant_chunks
    )

    # ========================================================
    # DISPLAY RETRIEVED PLANT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVED PLANT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBotanical Name: {botanical_name}"
    )

    print(
        f"Family: {family}"
    )

    print(
        f"Plant ID: {plant_id}"
    )

    print(
        f"Chunks found: {len(plant_chunks)}"
    )

    print(
        f"Medicinal chunks found: "
        f"{len(medicinal_chunks)}"
    )

    if medicinal_chunks:

        print(
            "Medicinal chunk indexes: "
            +
            ", ".join(
                str(
                    chunk.get(
                        "chunk_index",
                        "?"
                    )
                )
                for chunk in medicinal_chunks
            )
        )

    # ========================================================
    # STEP 4: EXTRACT THERAPEUTIC INFORMATION
    # ========================================================

    therapeutic_results = (
        extract_therapeutic_information(
            medicinal_chunks
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EXTRACTED THERAPEUTIC DATABASE INFORMATION"
    )

    print(
        "=" * 70
    )

    if therapeutic_results:

        for result in therapeutic_results:

            print(
                f"- {result}"
            )

    else:

        print(
            "No explicit therapeutic statements were extracted."
        )

    # ========================================================
    # STEP 5: FINAL ANSWER
    # ========================================================

    print(
        "\nGenerating final answer..."
    )

    answer = format_answer(
        plant,
        therapeutic_results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "MEDICINAL PLANT RAG SYSTEM"
    )

    print(
        "=" * 70
    )

    while True:

        question = input(
            "\nEnter your medicinal plant question: "
        ).strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "bye"
        }:

            print(
                "\nExiting..."
            )

            break

        answer_question(
            question
        )