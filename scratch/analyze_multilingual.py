import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_data():
    dev_path = PROJECT_ROOT / "medicinalplant_eval_questions.json"
    holdout_path = PROJECT_ROOT / "holdout_results.json"
    
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_items = json.load(f)
        
    with open(holdout_path, "r", encoding="utf-8") as f:
        holdout_data = json.load(f)
        holdout_items = holdout_data["results"] if "results" in holdout_data else holdout_data
        
    return dev_items, holdout_items

def analyze():
    dev_items, holdout_items = load_data()
    print("=== HOLDOUT MULTILINGUAL ITEMS ===")
    for item in holdout_items:
        if item.get("language") != "English" or item.get("answer_correctness", 1.0) < 1.0:
            print(f"ID: {item.get('id')} | Lang: {item.get('language')} | Type: {item.get('question_type')}")
            print(f"  Q: {item.get('question')}")
            print(f"  Exp Entity: {item.get('expected_entity')} | Intent: {item.get('expected_intent')}")
            print(f"  Ref: {item.get('baseline_answer')}")
            print(f"  RAG: {item.get('rag_answer')}")
            print(f"  Corr: {item.get('answer_correctness')} | Sim: {item.get('semantic_similarity')} | Faith: {item.get('faithfulness')} | LangMatch: {item.get('language_match', 1.0)}")
            print("-" * 60)

    print("\n=== DEV MULTILINGUAL ITEMS ===")
    for item in dev_items:
        if item.get("language") != "English":
            print(f"ID: {item.get('id')} | Lang: {item.get('language')} | Type: {item.get('question_type')}")
            print(f"  Q: {item.get('question')}")
            print(f"  Exp Entity: {item.get('expected_entity')} | Intent: {item.get('expected_intent')}")
            print(f"  Ref: {item.get('baseline_answer')}")
            print(f"  RAG: {item.get('rag_answer')}")
            print(f"  Corr: {item.get('answer_correctness')} | Sim: {item.get('semantic_similarity')} | Faith: {item.get('faithfulness')} | LangMatch: {item.get('language_match', 1.0)}")
            print("-" * 60)

if __name__ == "__main__":
    analyze()
