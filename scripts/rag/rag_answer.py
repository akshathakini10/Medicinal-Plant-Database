import sys
import os
import json
import re
import requests
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# Paths & Setup
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.entity_resolver import PlantEntityResolver, normalize_text

INDEX_FILE = PROJECT_ROOT / "data" / "vector_store" / "medicinal_plants.index"
METADATA_FILE = PROJECT_ROOT / "data" / "vector_store" / "metadata.json"

from src.config import OLLAMA_URL, OLLAMA_MODEL, EMBED_MODEL_NAME, TOP_K, OLLAMA_TIMEOUT

# ============================================================
# Load Resources
# ============================================================
print("Initializing RAG components...")

model = SentenceTransformer(EMBED_MODEL_NAME)
index = faiss.read_index(str(INDEX_FILE))

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

resolver = PlantEntityResolver(
    knowledge_base_path=str(PROJECT_ROOT / "data/integrated/medicinal_knowledge_base.json"),
    ayurvedic_identity_path=str(PROJECT_ROOT / "data/cleaned/ayurvedic_identity_clean.json"),
    shlokas_path=str(PROJECT_ROOT / "data/cleaned/shlokas_clean.json")
)

def clean_context_text(text: str) -> str:
    """Strip raw Devanagari, OCR pipes, and excessive whitespace to speed up LLM processing."""
    if not text:
        return ""
    # Remove Devanagari Unicode block
    text = re.sub(r'[\u0900-\u097F]+', '', text)
    # Remove OCR tags and pipes
    text = re.sub(r'(\(Dhn\)|\(Bpn\)|\(Mdn\)|\(Rjn\)|\(Cki\)|\bIl\b|\bII\b|\bI\b|\|)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_shloka(d: dict) -> str:
    for key in ["english_explanation", "ocr_text", "shloka_text", "shlokas", "medicinal_information", "medicinal_uses", "discussion"]:
        val = d.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 3:
            return clean_context_text(val)
        elif val and isinstance(val, list):
            joined = " ".join([str(x).strip() for x in val if str(x).strip()])
            if len(joined) > 3:
                return clean_context_text(joined)
    return ""

def retrieve_chunks(query: str, top_k: int = TOP_K):
    """Retrieve top semantic context chunks from FAISS."""
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_vec.astype(np.float32), top_k)
    
    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata):
            retrieved.append(metadata[idx]["text"])
    return retrieved

def generate_with_ollama(user_query: str, context: str, structured_facts: str = "") -> str:
    system_prompt = (
        "You are an expert Ayurvedic and Botanical Assistant. "
        "Answer the user query strictly using the provided database facts and context.\n"
        "Rules:\n"
        "1. Identify the correct botanical name and family first.\n"
        "2. List therapeutic uses in concise bullet points.\n"
        "3. Translate Sanskrit disease terms (Kustha -> skin diseases, Jvara -> fever, Krmi -> worms, Sopha -> inflammation, Arsa -> piles/hemorrhoids, Pandu -> anemia).\n"
        "4. Do NOT output Sanskrit Devanagari text."
    )

    prompt_body = (
        f"{structured_facts}\n\n"
        f"Context Details:\n{context}\n\n"
        f"User Query: {user_query}\n\n"
        f"Answer:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_body,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 250
        }
    }

    try:
        print("[Gemma 3 generating...]", end="", flush=True)
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        print("\r" + " " * 28 + "\r", end="", flush=True)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception:
        print("\r" + " " * 28 + "\r", end="", flush=True)

    # Clean fallback if Ollama fails or times out
    return f"{structured_facts}\n\nMedicinal uses & Excerpts:\n- " + clean_context_text(context)[:400]

def handle_query(query: str) -> str:
    resolution = resolver.resolve_entity(query)
    structured_facts = ""
    target_plant_id = None
    direct_excerpts = []

    # 1. Exact entity resolved
    if resolution and resolution.get("records"):
        rec = resolution["records"][0]
        target_plant_id = rec.get("plant_id")
        
        botanical = rec.get("botanical_name", "N/A")
        family = rec.get("family", "N/A")
        systems = rec.get("systems", "N/A")
        matched = resolution.get("matched_term", "").title()

        structured_facts = (
            f"Matched Entity: {matched}\n"
            f"Botanical Name: {botanical}\n"
            f"Family: {family}\n"
            f"Systems of Medicine: {systems}"
        )

        # Collect verified shlokas from this exact plant record
        for s in rec.get("shlokas", []):
            t = extract_text_from_shloka(s)
            if t and t not in direct_excerpts:
                direct_excerpts.append(t)

        for s in resolution.get("shlokas", []):
            t = extract_text_from_shloka(s)
            if t and t not in direct_excerpts:
                direct_excerpts.append(t)

    # 2. Vector Context (filter by plant_id if entity is known)
    vector_chunks = retrieve_chunks(query, top_k=TOP_K)
    cleaned_chunks = []
    for c in vector_chunks:
        if target_plant_id:
            if f"Plant ID: {target_plant_id}" in c:
                cleaned_chunks.append(clean_context_text(c))
        else:
            cleaned_chunks.append(clean_context_text(c))

    all_context = "\n".join(direct_excerpts + cleaned_chunks)

    # 3. LLM Synthesis
    return generate_with_ollama(query, all_context, structured_facts)

if __name__ == "__main__":
    print("=" * 60)
    print("Medicinal Plant RAG Engine (Entity-Anchored + Gemma 3 12B)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Ask a plant question (or 'exit'): ").strip()
            if not user_input or user_input.lower() == "exit":
                break
            print("\n" + handle_query(user_input))
            print("\n" + "-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break