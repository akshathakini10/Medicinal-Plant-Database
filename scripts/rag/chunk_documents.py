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
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def read_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_metadata(text):
    plant_id_match = re.search(r"Plant ID:\s*(.+)", text)
    botanical_match = re.search(r"Botanical Name:\s*(.+)", text)
    family_match = re.search(r"Family:\s*(.+)", text)

    return {
        "plant_id": plant_id_match.group(1).strip() if plant_id_match else "",
        "botanical_name": botanical_match.group(1).strip() if botanical_match else "",
        "family": family_match.group(1).strip() if family_match else ""
    }

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            if len(paragraph) > chunk_size:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunks.append(paragraph[start:end].strip())
                    start = end - overlap
                current_chunk = ""
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_chunks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = list(INPUT_DIR.glob("*.txt"))
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

            # Metadata header prepended to every chunk so context is never lost
            header = (
                f"Plant ID: {metadata['plant_id']}\n"
                f"Botanical Name: {metadata['botanical_name']}\n"
                f"Family: {metadata['family']}\n\n"
            )

            for index, chunk in enumerate(chunks):
                # Don't duplicate header if it's already the very first chunk header
                chunk_text = chunk if index == 0 and chunk.startswith("MEDICINAL PLANT") else header + chunk

                chunk_record = {
                    "chunk_id": f"{metadata['plant_id']}_{index}",
                    "plant_id": metadata["plant_id"],
                    "botanical_name": metadata["botanical_name"],
                    "family": metadata["family"],
                    "chunk_index": index,
                    "text": chunk_text
                }

                all_chunks.append(chunk_record)
                total_chunks += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            skipped += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("DOCUMENT CHUNKING COMPLETED")
    print("=" * 60)
    print(f"Documents processed : {len(documents) - skipped}")
    print(f"Total chunks        : {total_chunks}")
    print(f"Saved to            : {OUTPUT_FILE}")

if __name__ == "__main__":
    create_chunks()