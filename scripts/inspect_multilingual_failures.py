import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def analyze_dataset(file_path, name):
    print("=" * 80)
    print(f"ANALYZING MULTILINGUAL FAILURES & LOW-SCORING ITEMS IN: {name}")
    print("=" * 80)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        items = data["results"]
        summary = data.get("summary", {})
        telemetry = data.get("telemetry", {})
        print("Summary:", summary)
        print("Telemetry:", telemetry)
    else:
        items = data

    multilingual_items = [item for item in items if item.get("language") != "English"]
    print(f"\nTotal items: {len(items)}, Multilingual items: {len(multilingual_items)}")

    for item in multilingual_items:
        qid = item.get("id")
        q = item.get("question")
        lang = item.get("language")
        qtype = item.get("question_type")
        ref = item.get("baseline_answer")
        rag = item.get("rag_answer")
        corr = item.get("answer_correctness", 0.0)
        sim = item.get("semantic_similarity", 0.0)
        faith = item.get("faithfulness", 0.0)
        lmatch = item.get("language_match", 1.0)
        exp_ent = item.get("expected_entity", "")
        intent = item.get("expected_intent", "")

        print(f"\n[ID {qid}] ({lang}) {qtype}")
        print(f"Q: {q}")
        print(f"Expected Entity: {exp_ent} | Intent: {intent}")
        print(f"Reference Answer: {ref}")
        print(f"RAG Answer: {rag}")
        print(f"Metrics -> Correctness: {corr:.4f} | SemSim: {sim:.4f} | Faith: {faith:.4f} | LangMatch: {lmatch}")

if __name__ == "__main__":
    analyze_dataset(PROJECT_ROOT / "holdout_results.json", "HOLDOUT BENCHMARK (20 Qs)")
    analyze_dataset(PROJECT_ROOT / "medicinalplant_eval_questions.json", "DEVELOPMENT BENCHMARK (60 Qs)")
