import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "rag_chunks" / "chunks.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_embeddings"
OUTPUT_EMBEDDINGS = OUTPUT_DIR / "embeddings.npy"
OUTPUT_METADATA = OUTPUT_DIR / "metadata.json"


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    print("Loading chunks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks loaded: {len(chunks)}")

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    print("\nLoading embedding model...")
    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("\nCreating embeddings...")
    print(f"Texts to encode: {len(texts)}")
    print(f"Batch size: {BATCH_SIZE}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS AND METADATA
# ============================================================

def save_embeddings(chunks, embeddings):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_EMBEDDINGS, embeddings)

    metadata = []

    for chunk in chunks:
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "plant_id": chunk["plant_id"],
            "botanical_name": chunk["botanical_name"],
            "family": chunk["family"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"]
        })

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n============================================================")
    print("EMBEDDING CREATION COMPLETED")
    print("============================================================")
    print(f"Chunks embedded : {len(chunks)}")
    print(f"Embedding shape : {embeddings.shape}")
    print(f"Embeddings saved:")
    print(OUTPUT_EMBEDDINGS)
    print(f"\nMetadata saved:")
    print(OUTPUT_METADATA)


# ============================================================
# MAIN
# ============================================================

def main():

    chunks = load_chunks()

    embeddings = create_embeddings(chunks)

    save_embeddings(chunks, embeddings)


if __name__ == "__main__":
    main()