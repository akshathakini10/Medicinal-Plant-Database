import json
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_EMBEDDINGS = (
    PROJECT_ROOT
    / "data"
    / "rag_embeddings"
    / "embeddings.npy"
)

INPUT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "rag_embeddings"
    / "metadata.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "vector_store"

OUTPUT_INDEX = OUTPUT_DIR / "medicinal_plants.index"
OUTPUT_METADATA = OUTPUT_DIR / "metadata.json"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print("Loading embeddings...")

    embeddings = np.load(INPUT_EMBEDDINGS)

    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    print("Loading metadata...")

    with open(INPUT_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Metadata records: {len(metadata)}")

    return metadata


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_index(embeddings):

    dimension = embeddings.shape[1]

    print("\nCreating FAISS index...")
    print(f"Vector dimension: {dimension}")

    # Inner Product works with normalized embeddings
    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype(np.float32)
    )

    print(f"Vectors indexed: {index.ntotal}")

    return index


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(index, metadata):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(OUTPUT_INDEX)
    )

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n============================================================")
    print("FAISS INDEX CREATION COMPLETED")
    print("============================================================")

    print(f"Vectors indexed : {index.ntotal}")
    print(f"Index dimension : {index.d}")
    print("\nFAISS index saved:")
    print(OUTPUT_INDEX)

    print("\nMetadata saved:")
    print(OUTPUT_METADATA)


# ============================================================
# MAIN
# ============================================================

def main():

    embeddings = load_embeddings()

    metadata = load_metadata()

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Number of embeddings does not match "
            "number of metadata records."
        )

    index = create_index(embeddings)

    save_index(index, metadata)


if __name__ == "__main__":
    main()