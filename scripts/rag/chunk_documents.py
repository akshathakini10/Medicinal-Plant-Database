import re
import json
from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "rag_documents"
OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_chunks"

OUTPUT_FILE = OUTPUT_DIR / "chunks.json"


# ==========================================================
# Chunk Settings
# ==========================================================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ==========================================================
# Read Document
# ==========================================================

def read_document(file_path):
    """Read a RAG document using UTF-8."""

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================================
# Extract Plant Information
# ==========================================================

def extract_metadata(text):
    """Extract basic plant metadata from the document."""

    plant_id_match = re.search(
        r"Plant ID:\s*(.+)",
        text
    )

    botanical_match = re.search(
        r"Botanical Name:\s*(.+)",
        text
    )

    family_match = re.search(
        r"Family:\s*(.+)",
        text
    )

    return {
        "plant_id": plant_id_match.group(1).strip()
        if plant_id_match else "",

        "botanical_name": botanical_match.group(1).strip()
        if botanical_match else "",

        "family": family_match.group(1).strip()
        if family_match else ""
    }


# ==========================================================
# Split Text
# ==========================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split document into overlapping chunks.

    Paragraphs are kept together where possible.
    """

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If adding paragraph stays within chunk size
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:

            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            # If paragraph itself is larger than chunk size,
            # split it using character windows.
            if len(paragraph) > chunk_size:

                start = 0

                while start < len(paragraph):

                    end = start + chunk_size

                    piece = paragraph[start:end]

                    chunks.append(piece.strip())

                    start = end - overlap

                current_chunk = ""

            else:

                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ==========================================================
# Create Chunk Records
# ==========================================================

def create_chunks():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = list(
        INPUT_DIR.glob("*.txt")
    )

    print(f"Documents found: {len(documents)}")

    all_chunks = []

    total_chunks = 0

    skipped = 0

    for file_path in documents:

        try:

            text = read_document(file_path)

            if not text.strip():
                skipped += 1
                continue

            metadata = extract_metadata(text)

            chunks = split_text(text)

            for index, chunk in enumerate(chunks):

                chunk_record = {

                    "chunk_id":
                        f"{metadata['plant_id']}_{index}",

                    "plant_id":
                        metadata["plant_id"],

                    "botanical_name":
                        metadata["botanical_name"],

                    "family":
                        metadata["family"],

                    "chunk_index":
                        index,

                    "text":
                        chunk
                }

                all_chunks.append(
                    chunk_record
                )

                total_chunks += 1

        except Exception as e:

            print(
                f"Error processing "
                f"{file_path.name}: {e}"
            )

            skipped += 1


    # ======================================================
    # Save Chunks
    # ======================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ======================================================
    # Report
    # ======================================================

    print("\n" + "=" * 60)
    print("DOCUMENT CHUNKING COMPLETED")
    print("=" * 60)

    print(
        f"Documents processed : "
        f"{len(documents) - skipped}"
    )

    print(
        f"Documents skipped   : "
        f"{skipped}"
    )

    print(
        f"Total chunks        : "
        f"{total_chunks}"
    )

    print(
        f"Chunk size          : "
        f"{CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap       : "
        f"{CHUNK_OVERLAP}"
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    create_chunks()