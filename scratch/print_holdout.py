import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "holdout_results.json", "r", encoding="utf-8") as f:
    holdout_data = json.load(f)
    results = holdout_data["results"]

print(f"Total Holdout Items: {len(results)}")
for item in results:
    qid = item["id"]
    lang = item["language"]
    qtype = item["question_type"]
    q = item["question"]
    ref = item["baseline_answer"]
    rag = item["rag_answer"]
    corr = item["answer_correctness"]
    sim = item["semantic_similarity"]
    rec = item.get("token_recall", 0)
    print(f"--- [Holdout ID {qid:02d}] ({lang}) {qtype} ---")
    print(f"  Q:   {q}")
    print(f"  Ref: {ref}")
    print(f"  RAG: {rag}")
    print(f"  Corr: {corr:.4f} | Sim: {sim:.4f} | TokRec: {rec:.4f}")
