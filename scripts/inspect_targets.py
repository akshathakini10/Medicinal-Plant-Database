import json

with open("failure_analysis.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

rec_map = {r["test_id"]: r for r in records}
targets = [17, 22, 27, 34, 57]

for tid in targets:
    r = rec_map.get(tid, {})
    print("=" * 60)
    print(f"Test ID: {tid} ({r.get('language')})")
    print(f"Question: {r.get('question')}")
    print(f"Baseline: {r.get('baseline_answer')}")
    print(f"RAG Ans : {r.get('rag_answer')}")
    print(f"Corr: {r.get('answer_correctness')} | Labels: {r.get('failure_labels')}")
