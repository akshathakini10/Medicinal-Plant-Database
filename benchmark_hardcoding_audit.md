# Benchmark Hardcoding & Rule Classification Audit Report

**Date:** October 3, 2026  
**Audited File:** `src/graph_rag.py`  
**Scope:** Inspect all conditional branches, query handlers, and return statements for benchmark-specific overfitting vs general domain heuristics.

---

## 1. Executive Summary

An exhaustive line-by-line audit of `src/graph_rag.py` revealed that while the system contains valid general domain rules (such as property token protection and NetworkX graph traversal), **the majority of final answer generation branches in `GraphRAGService.query()` are hardcoded for specific questions in the 60-item development benchmark.**

This explains why the system scored **99.98% Answer Correctness** and **0.9997 Semantic Similarity** on the 60-question development benchmark, while dropping to **59.44% Answer Correctness** on the unseen 20-question holdout set.

---

## 2. Rule Classification Inventory

Each rule and branch in `src/graph_rag.py` is categorized below as either:
- **`GENERIC_DOMAIN_RULE`**: An architectural heuristic or data structure that generalizes to unseen plants, diseases, or queries.
- **`BENCHMARK_SPECIFIC_RULE`**: A hardcoded condition or return string tailored to pass a specific question in the 60-question development set.

### Table 1: Classification of Rules in `src/graph_rag.py`

| Code Location | Condition / Logic | Returned Output / Behavior | Classification | Target Benchmark Item |
| :--- | :--- | :--- | :---: | :---: |
| **Lines 48–56** | `RESERVED_PROPERTY_TOKENS` (`rasa`, `virya`, `vipaka`, `guna`, `taste`, `potency`, etc.) | Prevents property words from resolving as plant entities (e.g. *Commiphora myrrha*). | **`GENERIC_DOMAIN_RULE`** | General domain protection |
| **Lines 58–133** | `DISEASE_REGISTRY` | Maps Sanskrit disease names to English/vernacular aliases (*Prameha*, *Jvara*, *Arsa*, etc.). | **`GENERIC_DOMAIN_RULE`** | General domain knowledge |
| **Lines 135–151** | `CLASSICAL_ALIASES` | Maps Sanskrit plant names (*Guduchi*, *Ashoka*, *Nimba*, *Bilva*) to botanical binomials. | **`GENERIC_DOMAIN_RULE`** | General domain knowledge |
| **Lines 216–247** | `detect_language(query)` | Detects Devanagari script, Kannada script, Latin script, and code-mixed patterns. | **`GENERIC_DOMAIN_RULE`** | General multilingual handling |
| **Lines 250–276** | `is_disease_to_plant_query(query)` | Regular expression detection of reverse disease-to-plant lookup intent. | **`GENERIC_DOMAIN_RULE`** | General intent routing |
| **Lines 676–681** | `is_out_of_domain` checks for `paracetamol`, `metformin`, `insulin` | Returns modern pharmaceutical refusal messages. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q48, Q49 |
| **Lines 684–689** | `is_unknown_entity_query` checks for `XYZHerb`, `ABCDisease` | Refusal for synthetic unknown test names. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q46, Q47 |
| **Lines 701–702** | `res_type == "property_reverse_lookup"` | Returns: `"Plants sharing the same {virya_val} Virya as Guduchi include Citraka (Plumbago zeylanica) and Pippali."` | **`BENCHMARK_SPECIFIC_RULE`** | Hardcoded plant string for Q26 |
| **Lines 706–707** | `if "both a disease" in query_lower and "rasa" in query_lower:` | Returns: `"Guduchi (Tinospora cordifolia) is associated with documented disease relationships (Prameha, Jvara) and documented Rasa (Tikta)."` | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q28 |
| **Lines 710–711** | `if "connected with prameha" in query_lower and "ayurvedic properties" in query_lower:` | Returns: `"Guduchi (Tinospora cordifolia) is connected with Prameha in the database and has documented Ayurvedic properties (Tikta Rasa, Usna Virya)."` | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q31 |
| **Lines 714–715** | `if "prameha ge sambandhisida plants" in query_lower:` | Returns: `"Prameha ge sambandhisida plants featuring documented Ayurvedic properties include Guduchi (Tinospora cordifolia, Tikta Rasa/Usna Virya) and Acacia nilotica."` | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q59 |
| **Lines 719–720** | `if "common between" in query_lower and "prameha" in query_lower and "jvara" in query_lower:` | Returns: `"Medicinal plants common to both Prameha and Jvara include Guduchi (Tinospora cordifolia), Acacia catechu, and Justicia adhatoda."` | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q23 |
| **Lines 726–729** | `if "guduchi" in query_lower and "ashoka" in query_lower:` | Returns hardcoded comparisons for disease relationships (Arsa) and properties (Tikta/Usna vs Kashaya/Sitala). | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q24, Q25 |
| **Lines 733–734** | `if "disease relationships similar to" in query_lower and "ashoka" in query_lower:` | Returns: `"Plants with disease relationships similar to Ashoka (treating Arsa, Sopha, Vrana) include Guduchi (Tinospora cordifolia) and Citraka (Plumbago zeylanica)."` | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q27 |
| **Lines 742–745** | `if lang == "Hindi": return f"प्रमेह से संबंधित मुख्य औषधीय पौधे..."` | Returns hardcoded Hindi Prameha response mentioning Guduchi, Babool, Khair. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q52 |
| **Lines 744–746** | `if lang == "Kannada": return f"ಪ್ರಮೇಹಕ್ಕೆ (Prameha) ಸಂಬಂಧಿಸಿದ ಮುಖ್ಯ..."` | Returns hardcoded Kannada Prameha response mentioning Guduchi, Babool, Khadir. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q56 |
| **Lines 748–754** | `if "prameh" in query_lower and "linked" in query_lower:` / `"connected"` / `"similar"` | Specific string returns for spelling variations of Prameha. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q37, Q42, Q43 |
| **Lines 757–761** | `elif disease_name == "Arsa":` / `"Jvara":` / `"Kamala":` | Hardcoded plant lists for specific benchmark diseases. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q13, Q15, Q17 |
| **Lines 780–783** | `if "same plant as" in query_lower:` / `"related to" in query_lower:` | Returns exact identity confirmation for Guduchi and Ashoka. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q18, Q19 |
| **Lines 784–789** | `if "every disease" in query_lower:` / `"only information that exists":` / `"besides disease":` | Returns exact strings matching benchmark baseline answers. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q45, Q44, Q29 |
| **Lines 795–807** | `if "botanical name" in query_lower:` | Returns `"The botanical name of Ashoka is {b_name} (family {family})."` — hardcoded with the word "Ashoka"! | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q2, failed on holdout Q1 |
| **Lines 810–811** | `if "which medicinal plant is" in query_lower:` | Returns: `"Tinospora cordifolia corresponds to the Ayurvedic medicinal plant Guduchi (Giloy / Amrita)."` | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q3, failed on holdout Q3 |
| **Lines 814–815** | `if "family" in query_lower:` | Returns: `"Kutki ({b_name}) belongs to the family {family}."` — hardcoded with the word "Kutki"! | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q4, failed on holdout Q2 |
| **Lines 820–826** | `if "other names" in query_lower:` / `"synonym":` / `"referring to":` / `"resolve the name":` | Returns hardcoded Guduchi synonym lists. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q5, Q6, Q41, Q51 |
| **Lines 829–851** | `if "virya" in query_lower:` / `"rasa":` / `"vipaka":` | Returns hardcoded strings for Guduchi Virya, Ashoka Rasa, Kutki Vipaka. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q7, Q8, Q9, Q10 |
| **Lines 860–863** | `if "in the graph" in query_lower:` | Returns: `"Diseases associated with Guduchi in the graph include Prameha..."` vs `"documented disease relationships with"`. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q22 vs Q11/Q14 |
| **Lines 872–877** | `if "tell me about" in query_lower:` | Returns 1-sentence herb summary if query ends with `!` or contains `GUDUCHI`, else verbose summary. | **`BENCHMARK_SPECIFIC_RULE`** | Overfitted to Q1, Q33, Q34, Q35 |

---

## 3. Detailed Analysis of Entity-Specific Hardcoding

### 3.1 The "Ashoka" Hardcoding in Botanical Lookups
In `src/graph_rag.py` line 807:
```python
else:
    return f"The botanical name of Ashoka is {b_name} (family {family})."
```
When evaluated on the development benchmark (Q2: "What is the botanical name of Ashoka?"), this produced a perfect answer.  
However, when evaluated on the holdout benchmark (Holdout Q1: "What is the botanical name of Nimba?"), the system resolved *Nimba* correctly to *Azadirachta indica A. JUSS.*, but generated:
> *"The botanical name of Ashoka is Azadirachta indica A. JUSS. (family Meliaceae)."*

This provides undeniable proof of benchmark-specific hardcoding.

### 3.2 The "Kutki" Hardcoding in Family Lookups
In `src/graph_rag.py` line 815:
```python
if "family" in query_lower and "properties" not in query_lower and "rasa" not in query_lower:
    return f"Kutki ({b_name}) belongs to the family {family}."
```
When evaluated on Holdout Q2 ("Which family does Bilva belong to?"), the system resolved *Bilva* (*Aegle marmelos*) correctly, but output:
> *"Kutki (Aegle marmelos (L.) CORREA E. SCHULTZ) belongs to the family Rutaceae."*

### 3.3 The "Tinospora cordifolia" Hardcoding in Botanical-to-Common Lookups
In `src/graph_rag.py` line 811:
```python
if "which medicinal plant is" in query_lower:
    return "Tinospora cordifolia corresponds to the Ayurvedic medicinal plant Guduchi (Giloy / Amrita)."
```
When evaluated on Holdout Q3 ("Which medicinal plant is Acorus calamus?"), the system returned:
> *"Tinospora cordifolia corresponds to the Ayurvedic medicinal plant Guduchi (Giloy / Amrita)."*

---

## 4. Conclusion & Scientific Recommendations

1. **Root Cause of High Development Metrics:** The 99.98% correctness on the 60-question benchmark was achieved through comprehensive benchmark-specific rules (`BENCHMARK_SPECIFIC_RULE`) rather than general dynamic synthesis.
2. **Path to Generalization:**
   - Remove hardcoded plant and disease strings from response generators.
   - Replace deterministic entity-specific branches with dynamic structured templates:
     `f"The botanical name of {common_name} is {botanical_name} (family {family})."`
   - Utilize dynamic graph traversals for comparisons, intersections, and property reverse lookups rather than static plant pairs.
