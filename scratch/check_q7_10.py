import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('holdout_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)['results']

for x in results:
    if x['id'] in [7, 8, 9, 10]:
        print(f"--- Q{x['id']}: {x['question']} ---")
        print(f"Ref: {x['baseline_answer']}")
        print(f"RAG: {x['rag_answer']}")
        print(f"Corr: {x['answer_correctness']} | Sim: {x['semantic_similarity']} | TokRec: {x['token_recall']} | Faith: {x['faithfulness']}\n")
