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

# Number of results finally displayed
TOP_K = 5

# Retrieve more candidates from FAISS first
SEARCH_K = 50

# Weight given to exact keyword matches
EXACT_MATCH_BOOST = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)

print(f"Vectors in index: {index.ntotal}")


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

print(f"Metadata records: {len(metadata)}")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for exact keyword matching.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Remove punctuation but preserve Unicode letters
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Normalize multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# EXTRACT QUERY TERMS
# ============================================================

def get_query_terms(query):
    """
    Extract useful terms from the user's query.

    Example:
        What are the medicinal uses of Agaru?

    The important term becomes:
        agaru
    """

    normalized = normalize_text(query)

    stop_words = {
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
        "an"
    }

    terms = [
        word
        for word in normalized.split()
        if word not in stop_words
    ]

    return terms


# ============================================================
# EXACT MATCH SCORE
# ============================================================

def calculate_exact_match_score(query, result):
    """
    Give additional score when the query matches:

    1. Botanical name
    2. Drug name
    3. Sanskrit name
    4. Transliteration
    5. Other text in the chunk
    """

    query_normalized = normalize_text(query)

    query_terms = get_query_terms(query)

    botanical_name = normalize_text(
        result.get("botanical_name", "")
    )

    text = normalize_text(
        result.get("text", "")
    )

    # --------------------------------------------------------
    # Strong match: botanical name
    # --------------------------------------------------------

    if query_normalized == botanical_name:
        return 1.0

    # --------------------------------------------------------
    # Check drug_name / Sanskrit / transliteration
    # --------------------------------------------------------

    strong_fields = []

    for field in [
        "drug_name",
        "sanskrit_name",
        "transliteration"
    ]:

        value = result.get(field)

        if value:
            strong_fields.append(
                normalize_text(value)
            )

    # --------------------------------------------------------
    # Exact query in strong fields
    # --------------------------------------------------------

    for field_value in strong_fields:

        if query_normalized == field_value:
            return 1.0

        if query_normalized in field_value:
            return 0.9

    # --------------------------------------------------------
    # Look for drug_name / transliteration in chunk text
    # --------------------------------------------------------

    for term in query_terms:

        if f"drug name {term}" in text:
            return 0.95

        if f"transliteration {term}" in text:
            return 0.95

    # --------------------------------------------------------
    # Exact query anywhere in chunk
    # --------------------------------------------------------

    if query_normalized and query_normalized in text:
        return 0.70

    # --------------------------------------------------------
    # Individual query-term matching
    # --------------------------------------------------------

    if query_terms:

        matched_terms = sum(
            1
            for term in query_terms
            if term in text
        )

        return (
            matched_terms
            / len(query_terms)
            * 0.50
        )

    return 0.0


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search(query, top_k=TOP_K):

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------
    # FAISS semantic search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding.astype("float32"),
        SEARCH_K
    )

    # --------------------------------------------------------
    # Rerank results
    # --------------------------------------------------------

    reranked_results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(metadata):
            continue

        result = metadata[idx]

        exact_score = calculate_exact_match_score(
            query,
            result
        )

        # Combined score
        final_score = (
            float(score)
            + EXACT_MATCH_BOOST * exact_score
        )

        reranked_results.append(
            {
                "original_score": float(score),
                "exact_score": exact_score,
                "final_score": final_score,
                "result": result
            }
        )

    # --------------------------------------------------------
    # Sort by combined score
    # --------------------------------------------------------

    reranked_results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Remove duplicate plants
    # --------------------------------------------------------

    unique_results = []
    seen_plants = set()

    for item in reranked_results:

        plant_id = item["result"]["plant_id"]

        if plant_id in seen_plants:
            continue

        seen_plants.add(plant_id)

        unique_results.append(item)

    # IMPORTANT:
    # This must be AFTER the for loop
    reranked_results = unique_results

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for rank, item in enumerate(
        reranked_results[:top_k],
        start=1
    ):

        result = item["result"]

        print(f"\n--- RESULT {rank} ---")

        print(
            f"FAISS Score      : "
            f"{item['original_score']:.4f}"
        )

        print(
            f"Exact Match Score: "
            f"{item['exact_score']:.4f}"
        )

        print(
            f"Final Score      : "
            f"{item['final_score']:.4f}"
        )

        print(
            f"Plant ID         : "
            f"{result['plant_id']}"
        )

        print(
            f"Botanical Name   : "
            f"{result['botanical_name']}"
        )

        print(
            f"Family           : "
            f"{result['family']}"
        )

        print(
            f"Chunk Index      : "
            f"{result['chunk_index']}"
        )

        print("\nTEXT:")
        print(
            result["text"][:1500]
        )


# ============================================================
# TEST QUERY
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your medicinal plant question: "
    )

    search(query)