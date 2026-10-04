# Ground-Truth & Evaluator Leakage Audit Report

**Date:** October 3, 2026  
**Audited System:** Multilingual Ayurvedic Hybrid Graph-RAG (`src/graph_rag.py`)  
**Evaluator Scripts:** `scripts/eval_hybrid_rag.py`, `scripts/generate_failure_analysis.py`  
**Benchmark Set:** `medicinalplant_eval_questions.json` (60 Frozen Questions)

---

## 1. Executive Summary

This independent audit investigated whether ground-truth benchmark information, expected test metadata, or question identifiers were leaked into the production RAG pipeline (`GraphRAGService`).

**Audit Verdict:**
- **Runtime Variable Leakage:** **NONE DETECTED.** At execution time, `rag_service.query(user_query)` receives only the user's natural language question string. Evaluator metadata (`baseline_answer`, `expected_entity`, `expected_intent`, `test_id`) are never passed into the retrieval or synthesis call stacks.
- **Architectural / Design-Time Overfitting:** **EXTENSIVE.** While variable references do not cross into `src/graph_rag.py`, the return strings and query routing conditions within `src/graph_rag.py` were heavily pre-programmed to match the exact phrasing of the 60 benchmark ground-truth answers.

---

## 2. Static Codebase Search for Leakage Tokens

A comprehensive regex search across `src/` was conducted for the following prohibited evaluation identifiers:
`baseline_answer`, `expected_answer`, `ground_truth`, `expected_entity`, `expected_intent`, `question_id`, `test_id`, `benchmark`.

### Results in `src/`:
```text
Search Query: baseline_answer|expected_answer|ground_truth|expected_entity|expected_intent|question_id|test_id|benchmark
Target Path: src/
Matches Found: 0
```
No evaluation metadata variables exist in `src/graph_rag.py`, `src/entity_resolver.py`, or `src/query_router.py`.

### Results in `scripts/`:
In the evaluation scripts (`eval_hybrid_rag.py`, `generate_failure_analysis.py`, `test_full_synthesis.py`), the variables appear strictly in post-generation comparison blocks:
```python
# Evaluator runtime flow in scripts/eval_hybrid_rag.py:
for item in questions:
    q_text = item["question"]
    # Production call receives ONLY question text:
    rag_ans = rag_service.query(q_text)
    
    # Post-hoc evaluation comparison:
    baseline_ans = item["baseline_answer"]
    emb_base = eval_embedder.encode(baseline_ans, convert_to_tensor=True)
    emb_rag = eval_embedder.encode(rag_ans, convert_to_tensor=True)
    sem_sim = float(util.cos_sim(emb_base, emb_rag)[0][0])
```

---

## 3. Verification Checklist

| Leakage Condition | Status | Audit Findings |
| :--- | :---: | :--- |
| `baseline_answer` passed into `GraphRAGService` | **PASSED** | Never passed as argument; `query(user_query)` accepts only string `user_query`. |
| `baseline_answer` included in LLM prompt | **PASSED** | LLM prompt contains only retrieved graph/FAISS context and question. |
| `baseline_answer` used during retrieval | **PASSED** | FAISS and NetworkX search solely on tokens extracted from `user_query`. |
| Test IDs used to choose answers | **PASSED** | No conditional checks on test ID (`qid`, `test_id`) exist in `src/`. |
| `expected_entity` used for entity resolution | **PASSED** | Resolver uses `HierarchicalResolver` regex and graph searches on `user_query`. |
| `expected_intent` used for query routing | **PASSED** | Router uses `detect_language`, property regexes, and graph edge traversal. |
| Evaluator-only metadata in production scope | **PASSED** | Evaluator variables are encapsulated in standalone test scripts. |

---

## 4. Distinction Between Runtime Leakage and Prompt/Rule Overfitting

While there is no **direct variable leakage**, the audit uncovered that the system exhibits **design-time overfitting**:
1. Specific query phrases from the development benchmark (e.g., `"both a disease" and "rasa"`, `"connected with prameha" and "ayurvedic properties"`, `"which medicinal plant is"`) trigger dedicated deterministic return strings in `src/graph_rag.py`.
2. These hardcoded return strings mirror the benchmark's `baseline_answer` text word-for-word.
3. Consequently, the pipeline achieves 0.9997 semantic similarity and 99.98% answer correctness on the 60 development questions not via dynamic generalization, but because the answers were baked into the service's decision tree.

The details and complete classification of these rules are documented in [`benchmark_hardcoding_audit.md`](file:///c:/Users/Akshatha/OneDrive/Desktop/MedicinalPlantScraper/benchmark_hardcoding_audit.md).
