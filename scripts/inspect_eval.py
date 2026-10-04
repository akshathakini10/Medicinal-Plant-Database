import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


with open("medicinalplant_eval_questions.json", "r", encoding="utf-8") as f:
    qs = json.load(f)

with open("failure_analysis.jsonl", "r", encoding="utf-8") as f:
    failures = [json.loads(line) for line in f]

fail_by_id = {r["test_id"]: r for r in failures}

print(f"Total questions: {len(qs)}")
from src.graph_rag import GraphRAGService
rag = GraphRAGService()
with open("data/integrated/medicinal_knowledge_base.json", "r", encoding="utf-8") as f:
    kb = json.load(f)

for p in kb:
    b = p.get("botanical_name", "").lower()
    s = str(p.get("ayurvedic_identity", "")).lower()
    v = str(p.get("vernacular_names", "")).lower()
    if any(k in b or k in s or k in v for k in ["tinospora cordifolia", "saraca asoca", "picrorhiza kurroa", "nelumbo nucifera"]):
        print(f"Plant: {p.get('botanical_name')} | ID: {p.get('plant_id')}")
        print(f"  Ayur: {p.get('ayurvedic_identity')}")
        print(f"  Rasa: {p.get('rasa')} | Virya: {p.get('virya')} | Vipaka: {p.get('vipaka')}")
        print(f"  Diseases: {p.get('diseases')}")
        print(f"  Synonyms: {p.get('synonyms')}")








