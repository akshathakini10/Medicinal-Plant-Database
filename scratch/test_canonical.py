import sys
import os
import json
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_rag import KB_FILE, HierarchicalResolver, get_binomial

resolver = HierarchicalResolver(KB_FILE)

def is_canonical_entity_match(expected_entity_str: str, resolved: dict, resolver) -> bool:
    if not expected_entity_str:
        return resolved.get("type") in ["out_of_domain", "unknown_plant", "unknown_disease"] or "unknown" in resolved.get("type", "")
    
    exp_lower = expected_entity_str.lower().strip()
    res_type = resolved.get("type", "")
    
    # 1. Out of domain / unknown handling
    if "out of domain" in exp_lower or "unsupported" in exp_lower or exp_lower == "none":
        return res_type in ["out_of_domain", "unknown_plant", "unknown_disease"] or "unknown" in res_type
    if "unknown" in exp_lower:
        return "unknown" in res_type or res_type == "out_of_domain"

    # 2. Multi-entity intersection / comparison queries (e.g. "Bilva and Nimba", "Kasa, Svasa")
    sub_entities = [s.strip() for s in re.split(r",\s*|\s+and\s+|\s+mattu\s+|\s+aur\s+", expected_entity_str) if s.strip()]
    if len(sub_entities) > 1:
        res_data = resolved.get("data")
        if isinstance(res_data, list):
            all_matched = True
            for sub in sub_entities:
                sub_matched = False
                sub_res = resolver.resolve_query(sub)
                sub_bot = get_binomial(sub_res.get("data", {}).get("botanical_name", "")) if isinstance(sub_res.get("data"), dict) else ""
                sub_dis = sub_res.get("data", {}).get("name", "").lower() if isinstance(sub_res.get("data"), dict) else ""
                for r_item in res_data:
                    r_str = str(r_item).lower()
                    if (sub_bot and sub_bot in r_str) or (sub_dis and sub_dis in r_str) or sub.lower() in r_str or r_str in sub.lower():
                        sub_matched = True
                        break
                if not sub_matched:
                    all_matched = False
                    break
            if all_matched:
                return True

    # 3. Direct canonical resolution of expected entity
    exp_res = resolver.resolve_query(expected_entity_str)
    exp_data = exp_res.get("data") if isinstance(exp_res.get("data"), dict) else {}
    res_data = resolved.get("data") if isinstance(resolved.get("data"), dict) else {}
    
    # Botanical species comparison
    exp_bot = exp_data.get("botanical_name", "")
    res_bot = res_data.get("botanical_name", "")
    if exp_bot and res_bot:
        if get_binomial(exp_bot) == get_binomial(res_bot):
            return True
        if get_binomial(exp_bot) in res_bot.lower() or get_binomial(res_bot) in exp_bot.lower():
            return True
            
    # Disease canonical comparison
    exp_dis = exp_data.get("name", "")
    res_dis = res_data.get("name", "")
    if exp_dis and res_dis:
        if exp_dis.lower() == res_dis.lower():
            return True

    # Check plant ID
    exp_pid = str(exp_data.get("plant_id", "")).strip()
    res_pid = str(res_data.get("plant_id", "")).strip()
    if exp_pid and res_pid and exp_pid == res_pid:
        return True

    # 4. Synonym / Classical alias check via resolved data
    pred_term = str(resolved.get("matched_term", "")).lower()
    res_terms = {pred_term, str(res_data.get("name", "")).lower(), str(res_data.get("botanical_name", "")).lower(), str(res_data.get("sanskrit_name", "")).lower()}
    for syn in res_data.get("synonyms", []):
        res_terms.add(str(syn).lower())
    for ident in res_data.get("ayurvedic_identity", []):
        if isinstance(ident, dict):
            res_terms.add(str(ident.get("drug_name", "")).lower())
            res_terms.add(str(ident.get("transliteration", "")).lower())
            
    exp_tokens = [t for t in re.findall(r"[\w\u0900-\u097F\u0C80-\u0CFF]+", exp_lower) if len(t) > 2]
    for term in res_terms:
        if term and any(t == term or t in term or term in t for t in exp_tokens):
            return True

    if exp_lower in pred_term or pred_term in exp_lower:
        return True

    return False

# Test against holdout questions
holdout_path = PROJECT_ROOT / "medicinalplant_holdout_eval.jsonl"
with open(holdout_path, "r", encoding="utf-8") as f:
    holdout = [json.loads(line) for line in f]

all_passed = True
for item in holdout:
    q = item["question"]
    exp = item["expected_entity"]
    res = resolver.resolve_query(q)
    match = is_canonical_entity_match(exp, res, resolver)
    print(f"[{item['id']}] Q: '{q[:35]}...' | Exp: '{exp}' | Res matched: '{res.get('matched_term')}' | Match: {match}")
    if not match:
        all_passed = False

print(f"\nAll holdout questions canonical entity match: {all_passed}")
