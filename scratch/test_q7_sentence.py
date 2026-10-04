import requests
import json
import re
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

MODEL = "gemma3:12b"
OLLAMA_URL = "http://localhost:11434/api/generate"

sys_p = """You are the multilingual response-generation layer for an Ayurvedic medicinal-plant Hybrid Graph-RAG system.
Synthesize verified evidence into an accurate, natural sentence that directly answers the question.

STRICT INSTRUCTIONS:
1. Always form a complete, fluent sentence answering the question (e.g. "Medicinal plants associated with Kasa include...").
2. NEVER output raw field labels or JSON keys like 'matching_plants:' or 'common_plants:' in your response.
3. Keep botanical Latin names in exact Latin script.
4. Answer in the same language and style as the user query (natural English for English, natural Hindi for Hindi, natural Kannada for Kannada, natural code-mix for code-mixed queries)."""

user_query = "Which medicinal plants are associated with Kasa?"
evidence_text = "- Disease Condition: Kasa (cough / bronchitis)\n- Documented Plants: Acacia nilotica, Acacia catechu, Abies pindrow, Abies spectabilis, Justicia adhatoda"

prompt = f"User Question: {user_query}\nLanguage Style: English\n\nLocked Facts:\n{evidence_text}\n\nAnswer:"

payload = {
    "model": MODEL,
    "system": sys_p,
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 75},
    "keep_alive": "30m"
}

resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
cand = resp.json().get("response", "").strip()
print("Generated candidate:\n", cand)

ref = "Medicinal plants associated with Kasa (cough / bronchitis) include Acacia nilotica, Acacia catechu, Abies pindrow, and Abies spectabilis."
emb_ref = embedder.encode(ref, convert_to_tensor=True)
emb_cand = embedder.encode(cand, convert_to_tensor=True)
sim = float(util.cos_sim(emb_ref, emb_cand)[0][0])

ref_words = set(re.findall(r"\w+", ref.lower())) - {"the", "a", "an", "is", "are", "of", "and", "in", "to", "for", "with"}
cand_words = set(re.findall(r"\w+", cand.lower())) - {"the", "a", "an", "is", "are", "of", "and", "in", "to", "for", "with"}
rec = len(ref_words.intersection(cand_words)) / len(ref_words)
corr = 0.6 * sim + 0.4 * rec
print(f"Sim: {sim:.4f} | TokRec: {rec:.4f} | Corr: {corr:.4f}")
