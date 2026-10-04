import sys
import os
import json
import csv
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_rag import (
    GraphRAGService, is_out_of_domain, is_unknown_entity_query,
    detect_language, is_canonical_entity_match,
    extract_full_evidence_context, calculate_faithfulness
)

HOLDOUT_FILE = PROJECT_ROOT / "medicinalplant_holdout_eval.jsonl"
OUT_JSON = PROJECT_ROOT / "holdout_results.json"
OUT_CSV = PROJECT_ROOT / "holdout_results.csv"

def calculate_token_recall(reference: str, candidate: str) -> float:
    ref_words = set(re.findall(r"\w+", reference.lower(), flags=re.UNICODE))
    cand_words = set(re.findall(r"\w+", candidate.lower(), flags=re.UNICODE))
    stop_words = {"the", "a", "an", "is", "are", "of", "and", "in", "to", "for", "with", "this", "that"}
    ref_words -= stop_words
    cand_words -= stop_words
    if not ref_words:
        return 1.0
    overlap = ref_words.intersection(cand_words)
    return len(overlap) / len(ref_words)

def main():
    use_llm = "--no-llm" not in sys.argv
    llm_mode_str = "HYBRID GRAPH-RAG + GEMMA 3 12B" if use_llm else "DETERMINISTIC GRAPH BASELINE"
    print("=" * 70)
    print(f"RUNNING INDEPENDENT HOLDOUT EVALUATION ({llm_mode_str})")
    print("=" * 70 + "\n")

    with open(HOLDOUT_FILE, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    print(f"Loaded {len(questions)} independent unseen holdout questions.")

    eval_embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    rag_service = GraphRAGService()

    results = []

    for item in questions:
        qid = item["id"]
        q_text = item["question"]
        q_type = item["question_type"]
        q_lang = item["language"]
        exp_entity = item["expected_entity"]
        exp_intent = item["expected_intent"]
        baseline_ans = item["baseline_answer"]

        # Run pipeline
        rag_ans = rag_service.query(q_text, use_llm=use_llm)

        # Entity resolution
        resolved = rag_service.resolver.resolve_query(q_text)
        is_ood = is_out_of_domain(q_text)
        unk_type, _ = is_unknown_entity_query(q_text)

        # 1. Entity Resolution Accuracy (Generic Canonical Botanical & Disease Evaluation)
        if is_ood or "out_of_domain" in q_type:
            entity_corr = 1.0 if is_ood else 0.0
            intent_corr = 1.0 if is_ood else 0.0
            retrieval_succ = 1.0
            context_rel = 1.0
            faithfulness = 1.0
        elif unk_type or "unknown" in q_type:
            entity_corr = 1.0 if unk_type else 0.0
            intent_corr = 1.0 if unk_type else 0.0
            retrieval_succ = 1.0
            context_rel = 1.0
            faithfulness = 1.0
        else:
            entity_corr = 1.0 if is_canonical_entity_match(exp_entity, resolved, rag_service.resolver) else 0.0
            intent_corr = 1.0 if resolved["type"] != "unknown" else 0.0
            retrieval_succ = 1.0 if entity_corr == 1.0 else 0.0
            context_rel = 1.0 if entity_corr == 1.0 else 0.5
            
            # Faithfulness (Groundedness against Full Evidence Context + Baseline)
            full_ctx = extract_full_evidence_context(rag_service, resolved, q_text)
            faithfulness = calculate_faithfulness(rag_ans, baseline_ans, full_ctx)

        # Semantic Similarity
        emb_base = eval_embedder.encode(baseline_ans, convert_to_tensor=True)
        emb_rag = eval_embedder.encode(rag_ans, convert_to_tensor=True)
        sem_sim = max(0.0, min(1.0, float(util.cos_sim(emb_base, emb_rag)[0][0])))

        # Token recall & Correctness
        token_rec = calculate_token_recall(baseline_ans, rag_ans)
        ans_corr = min(1.0, round(0.6 * sem_sim + 0.4 * token_rec, 4))

        # Language match
        det_lang = detect_language(rag_ans)
        if q_lang == "English":
            lang_match = 1.0 if det_lang == "English" else 0.0
        elif q_lang == "Hindi":
            lang_match = 1.0 if ("\u0900" <= max(rag_ans) <= "\u097F" or det_lang == "Hindi") else 0.0
        elif q_lang == "Kannada":
            lang_match = 1.0 if ("\u0C80" <= max(rag_ans) <= "\u0CFF" or det_lang == "Kannada") else 0.0
        else:
            lang_match = 1.0

        rec = {
            "id": qid,
            "question": q_text,
            "language": q_lang,
            "question_type": q_type,
            "expected_entity": exp_entity,
            "expected_intent": exp_intent,
            "baseline_answer": baseline_ans,
            "rag_answer": rag_ans,
            "entity_resolution_accuracy": entity_corr,
            "intent_accuracy": intent_corr,
            "retrieval_success": retrieval_succ,
            "context_relevance": context_rel,
            "faithfulness": faithfulness,
            "semantic_similarity": round(sem_sim, 4),
            "token_recall": round(token_rec, 4),
            "answer_correctness": ans_corr,
            "language_match": lang_match
        }
        results.append(rec)
        print(f"[{qid:02d}/20] ({q_lang}) '{q_text[:35]}...' -> Sim: {sem_sim:.4f} | Corr: {ans_corr:.4f} | Ent: {entity_corr}")

    # Summary
    summary = {
        "total_holdout_questions": len(results),
        "overall_answer_correctness": round(float(np.mean([r["answer_correctness"] for r in results])), 4),
        "average_semantic_similarity": round(float(np.mean([r["semantic_similarity"] for r in results])), 4),
        "entity_resolution_accuracy": round(float(np.mean([r["entity_resolution_accuracy"] for r in results])), 4),
        "intent_accuracy": round(float(np.mean([r["intent_accuracy"] for r in results])), 4),
        "retrieval_success_rate": round(float(np.mean([r["retrieval_success"] for r in results])), 4),
        "context_relevance": round(float(np.mean([r["context_relevance"] for r in results])), 4),
        "faithfulness": round(float(np.mean([r["faithfulness"] for r in results])), 4),
        "language_match_accuracy": round(float(np.mean([r["language_match"] for r in results])), 4),
    }

    # Multilingual & unsupported subsets
    multi_items = [r for r in results if r["language"] != "English"]
    summary["multilingual_correctness"] = round(float(np.mean([r["answer_correctness"] for r in multi_items])), 4) if multi_items else 1.0

    unsupp_items = [r for r in results if "out_of_domain" in r["question_type"] or "unknown" in r["question_type"]]
    summary["unsupported_query_handling"] = round(float(np.mean([r["answer_correctness"] for r in unsupp_items])), 4) if unsupp_items else 1.0

    print("\n" + "=" * 70)
    print("INDEPENDENT HOLDOUT EVALUATION SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"{k:<35}: {v}")
    print("=" * 70)
    print(f"Gemma 3 12B Telemetry:")
    for tk, tv in rag_service.telemetry.items():
        print(f"  {tk:<30}: {tv}")
    print("=" * 70)

    # Save to JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "telemetry": rag_service.telemetry, "results": results}, f, indent=2, ensure_ascii=False)

    # Save to CSV
    fieldnames = list(results[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved holdout results to:\n- {OUT_JSON}\n- {OUT_CSV}\n")

if __name__ == "__main__":
    main()
