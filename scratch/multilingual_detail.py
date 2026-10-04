import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "medicinalplant_eval_questions.json", "r", encoding="utf-8") as f:
    dev = json.load(f)

with open(PROJECT_ROOT / "holdout_results.json", "r", encoding="utf-8") as f:
    holdout = json.load(f)["results"]

print("=== DEV BENCHMARK MULTILINGUAL / LOW CORRECTNESS ===")
for q in dev:
    if q["language"] != "English":
        print(f"DEV-{q['id']:02d} | Lang: {q['language']} | Type: {q['question_type']}")
        print(f"  Q: {q['question']}")
        print(f"  Expected: {q['expected_entity']} | Intent: {q['expected_intent']}")
        print(f"  Ref: {q['baseline_answer']}")
        print(f"  RAG: {q.get('rag_answer', '')}")
        print(f"  Corr: {q.get('answer_correctness', 0):.4f} | Sim: {q.get('semantic_similarity', 0):.4f} | Faith: {q.get('faithfulness', 0):.4f}")
        print()

print("=== HOLDOUT BENCHMARK MULTILINGUAL / LOW CORRECTNESS ===")
for q in holdout:
    if q["language"] != "English":
        print(f"HOLD-{q['id']:02d} | Lang: {q['language']} | Type: {q['question_type']}")
        print(f"  Q: {q['question']}")
        print(f"  Expected: {q['expected_entity']} | Intent: {q['expected_intent']}")
        print(f"  Ref: {q['baseline_answer']}")
        print(f"  RAG: {q.get('rag_answer', '')}")
        print(f"  Corr: {q.get('answer_correctness', 0):.4f} | Sim: {q.get('semantic_similarity', 0):.4f} | Faith: {q.get('faithfulness', 0):.4f}")
        print()
