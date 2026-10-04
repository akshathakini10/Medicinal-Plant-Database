import pytest
import re
from src.graph_rag import (
    HierarchicalResolver, GraphRAGService, KB_FILE,
    is_universal_claim_query, extract_full_evidence_context, calculate_faithfulness,
    validate_groundedness, is_canonical_entity_match
)


# Create resolver and service once for tests
resolver = HierarchicalResolver(KB_FILE)
service = None

def get_service():
    global service
    if service is None:
        service = GraphRAGService()
    return service


# ============================================================
# Preserved Original Regression Tests
# ============================================================

def test_kamala_disease_intent():
    result = resolver.resolve_query(
        "Which medicinal plants treat Kamala?"
    )
    assert result["type"] == "disease"
    assert result["data"]["name"] == "Kamala"


def test_kamala_plant_intent():
    result = resolver.resolve_query(
        "Tell me about Kamala"
    )
    assert result["type"] == "plant"
    botanical_name = result["data"].get("botanical_name", "").lower()
    assert "nelumbo nucifera" in botanical_name


def test_prameha_reverse_lookup_intent():
    result = resolver.resolve_query(
        "Which medicinal plants treat Prameha?"
    )
    assert result["type"] == "disease"
    assert result["data"]["name"] == "Prameha"


def test_same_virya_as_guduchi():
    res = resolver.resolve_query(
        "Which plants share the same Virya as Guduchi?"
    )
    assert res["type"] == "property_reverse_lookup"
    assert "virya" in res.get("property", "").lower()

    rag = get_service()
    ans = rag.query("Which plants share the same Virya as Guduchi?", use_llm=False)
    assert any(term in ans.lower() for term in ["usna", "citraka", "plumbago", "pippali", "plants sharing"])


def test_property_tokens_do_not_resolve_as_plant():
    res_rasa = resolver.resolve_query("What is Rasa?")
    assert res_rasa.get("matched_term", "") != "Commiphora myrrha"
    assert res_rasa.get("matched_term", "").lower() != "rasa"

    res_props = resolver.resolve_query("Find a medicinal plant associated with both a disease relationship and documented Rasa information.")
    assert res_props["type"] == "plant"
    assert res_props.get("matched_term", "") != "Rasa"


def test_hindi_answer_remains_hindi():
    rag = get_service()
    ans = rag.query("अशोक का रस क्या है?", use_llm=False)
    assert re.search(r"[\u0900-\u097F]", ans) is not None
    assert "कषाय" in ans or "अशोक" in ans


def test_kannada_answer_remains_kannada():
    rag = get_service()
    ans = rag.query("ಅಶೋಕದ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು ಏನು?", use_llm=False)
    assert re.search(r"[\u0C80-\u0CFF]", ans) is not None
    assert "Saraca asoca" in ans or "ಸಸ್ಯಶಾಸ್ತ್ರೀಯ" in ans


def test_complete_disease_plant_enumeration_not_truncated():
    rag = get_service()
    plants = rag.get_plants_for_disease("Prameha")
    assert len(plants) > 8
    assert len(plants) >= 100
    b_names = [p.get("botanical_name", "").lower() for p in plants]
    assert any("tinospora cordifolia" in b for b in b_names)


def test_unknown_entity_handling():
    rag = get_service()
    ans = rag.query("Give me information about an unknown plant called XYZHerb.", use_llm=False)
    assert any(phrase in ans.lower() for phrase in ["could not identify", "unknown", "check the spelling", "xyzherb"])


def test_out_of_domain_handling():
    rag = get_service()
    ans1 = rag.query("What is the dosage of paracetamol?", use_llm=False)
    assert any(phrase in ans1.lower() for phrase in ["outside the scope", "out of domain", "does not provide", "medication"])

    ans2 = rag.query("Who is the Prime Minister of India?", use_llm=False)
    assert any(phrase in ans2.lower() for phrase in ["outside the scope", "out of domain", "specifically about"])


# ============================================================
# New Generic Generalization Tests (Entities NOT in Dev 60)
# ============================================================

def test_unseen_plant_botanical_lookup():
    rag = get_service()
    ans = rag.query("What is the botanical name of Nimba?", use_llm=False)
    assert "Azadirachta indica" in ans
    assert "Meliaceae" in ans
    # Must NOT mention Ashoka
    assert "Ashoka" not in ans


def test_unseen_plant_family_lookup():
    rag = get_service()
    ans = rag.query("Which family does Bilva belong to?", use_llm=False)
    assert "Rutaceae" in ans
    assert "Aegle marmelos" in ans
    # Must NOT mention Kutki
    assert "Kutki" not in ans


def test_unseen_property_lookup_vacha():
    rag = get_service()
    ans = rag.query("Give the Rasa, Virya and Vipaka of Vacha.", use_llm=False)
    assert "Acorus calamus" in ans or "Vacha" in ans
    assert "Usna" in ans
    assert "Katu" in ans


def test_unseen_disease_reverse_lookup_kasa():
    rag = get_service()
    ans = rag.query("Which medicinal plants are associated with Kasa?", use_llm=False)
    assert any(tax in ans for tax in ["Acacia nilotica", "Acacia catechu", "Abies pindrow", "Abies spectabilis"])


def test_unseen_disease_intersection_kasa_svasa():
    rag = get_service()
    ans = rag.query("Which plants are common between Kasa and Svasa?", use_llm=False)
    # Must dynamically find common plants between Kasa and Svasa from graph
    assert any(tax in ans for tax in ["Abies pindrow", "Abies spectabilis", "Abrus precatorius", "Carica papaya", "Peucedanum"])


def test_kannada_morphological_inflection_bilvada():
    rag = get_service()
    # Query with Kannada genitive suffix -da ("Bilvada")
    ans = rag.query("Bilvada virya enu?", use_llm=False)
    assert "Usna" in ans
    # Native Kannada script with inflection "ಬಿಲ್ವದ"
    ans_kn = rag.query("ಬಿಲ್ವದ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು ಏನು?", use_llm=False)
    assert "Aegle marmelos" in ans_kn


def test_unseen_novel_out_of_domain_ibuprofen():
    rag = get_service()
    ans = rag.query("What is the recommended dose of Ibuprofen for fever?", use_llm=False)
    assert any(phrase in ans.lower() for phrase in ["outside the scope", "out of domain", "medication"])


def test_unseen_unknown_medicinal_entity_fakiumherb():
    rag = get_service()
    ans = rag.query("Tell me about the medicinal uses of FakiumHerb.", use_llm=False)
    assert any(phrase in ans.lower() for phrase in ["could not identify", "fakiumherb", "check the spelling"])


def test_unseen_code_mixed_tulsi_family():
    rag = get_service()
    ans = rag.query("Tulsi kis family ko belong karti hai?", use_llm=False)
    assert "Lamiaceae" in ans
    assert "Ocimum" in ans or "Tulsi" in ans


# ============================================================
# Gemma 3 12B, Zero-Llama, Grounding & Canonical Evaluation Tests
# ============================================================

from unittest.mock import patch, MagicMock
from pathlib import Path
from src.config import OLLAMA_MODEL, OLLAMA_URL
from src.graph_rag import validate_groundedness, is_canonical_entity_match


def test_active_generation_model_is_gemma3_12b():
    assert OLLAMA_MODEL == "gemma3:12b"


def test_no_active_configuration_references_llama3():
    root = Path(__file__).resolve().parents[1]
    active_dirs = [root / "src", root / "scripts"]
    for d in active_dirs:
        for p in d.rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
            assert "llama3" not in text, f"Found reference to llama3 in active file: {p}"
            assert "llama3:latest" not in text, f"Found reference to llama3:latest in active file: {p}"


def test_gemma_invoked_for_synthesis_query():
    rag = get_service()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "The botanical name of Nimba is Azadirachta indica A. JUSS."
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        ans = rag.query("What is the botanical name of Nimba?", use_llm=True)
        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        payload = call_kwargs.get("json", {})
        assert payload.get("model") == "gemma3:12b"
        assert "Azadirachta indica" in payload.get("prompt", "")
        assert "Azadirachta indica" in ans


def test_deterministic_fallback_when_ollama_unavailable():
    rag = get_service()
    with patch("requests.post", side_effect=Exception("Ollama unreachable")):
        init_fallbacks = rag.telemetry.get("deterministic_fallbacks", 0)
        ans = rag.query("What is the botanical name of Nimba?", use_llm=True)
        # Must gracefully return deterministic answer without crashing
        assert "Azadirachta indica" in ans
        assert "Meliaceae" in ans
        assert rag.telemetry.get("deterministic_fallbacks", 0) > init_fallbacks


def test_hindi_gemma_response_remains_hindi():
    rag = get_service()
    ans = rag.query("नीम का वानस्पतिक नाम क्या है?", use_llm=False)
    assert re.search(r"[\u0900-\u097F]", ans) is not None
    assert "Azadirachta indica" in ans


def test_kannada_gemma_response_remains_kannada():
    rag = get_service()
    ans = rag.query("ಬಿಲ್ವದ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು ಏನು?", use_llm=False)
    assert re.search(r"[\u0C80-\u0CFF]", ans) is not None
    assert "Aegle marmelos" in ans


def test_groundedness_rejects_unsupported_disease_or_property():
    # 1. Contradictory virya: Plant has Usna virya, candidate claims Sita
    evidence = {"virya": "Usna", "botanical_name": "Aegle marmelos", "intent": "property_lookup"}
    candidate = "The Virya of Bilva is Sita (cool potency)."
    grounded, reason = validate_groundedness(candidate, evidence)
    assert not grounded
    assert "Virya contradictory" in reason

    # 2. Hallucinated disease: Candidate asserts plant treats a disease not in evidence
    evidence2 = {
        "diseases": ["Jvara", "Kasa"],
        "botanical_name": "Azadirachta indica",
        "intent": "plant_to_diseases",
        "user_query": "Which diseases does Nimba treat?"
    }
    candidate2 = "Nimba is widely used for treating Prameha and diabetes."
    grounded2, reason2 = validate_groundedness(candidate2, evidence2)
    assert not grounded2
    assert "Hallucinated disease" in reason2


def test_canonical_entity_evaluation_recognizes_synonyms():
    # Cross-lingual & classical synonym canonical recognition
    res_nimba = resolver.resolve_query("Neem")
    assert is_canonical_entity_match("Nimba", res_nimba, resolver)
    assert is_canonical_entity_match("नीम", res_nimba, resolver)
    assert is_canonical_entity_match("Azadirachta indica", res_nimba, resolver)

    res_bilva = resolver.resolve_query("ಬಿಲ್ವ")
    assert is_canonical_entity_match("Bilva", res_bilva, resolver)
    assert is_canonical_entity_match("Aegle marmelos", res_bilva, resolver)

    res_tulasi = resolver.resolve_query("Tulsi")
    assert is_canonical_entity_match("Tulasi", res_tulasi, resolver)
    assert is_canonical_entity_match("Ocimum tenuiflorum", res_tulasi, resolver)

    res_prameha = resolver.resolve_query("Diabetes")
    assert is_canonical_entity_match("Prameha", res_prameha, resolver)


# ============================================================
# Additional Comprehensive Regression Tests (Part 10 Requirements)
# ============================================================

def test_hindi_complex_property_answer_quality():
    rag = get_service()
    ans = rag.query("गुडूची का वानस्पतिक नाम और वीर्य बताइए।", use_llm=False)
    assert re.search(r"[\u0900-\u097F]", ans) is not None
    assert "Tinospora cordifolia" in ans
    assert any(term in ans for term in ["उष्ण", "Usna", "वीर्य"])


def test_kannada_complex_property_answer_quality():
    rag = get_service()
    ans = rag.query("ಗುಡುಚಿಯ ಆಯುರ್ವೇದ ಗುಣಗಳನ್ನು ತಿಳಿಸಿ.", use_llm=False)
    assert re.search(r"[\u0C80-\u0CFF]", ans) is not None
    assert any(term in ans for term in ["ತಿಕ್ತ", "Tikta", "ರಸ"])
    assert any(term in ans for term in ["ಉಷ್ಣ", "Usna", "ವೀರ್ಯ"])


def test_code_mixed_answer_fidelity():
    rag = get_service()
    ans_hi = rag.query("Tulsi kis family ko belong karti hai?", use_llm=False)
    assert "Lamiaceae" in ans_hi
    assert any(term in ans_hi.lower() for term in ["family", "belong", "karti hai", "tulsi"])

    ans_kn = rag.query("Bilvada virya enu?", use_llm=False)
    assert "Usna" in ans_kn
    assert any(term in ans_kn.lower() for term in ["bilva", "bilvada", "virya", "aagide"])


def test_botanical_latin_name_preservation():
    rag = get_service()
    ans_hi = rag.query("नीम का वानस्पतिक नाम क्या है?", use_llm=False)
    assert "Azadirachta indica" in ans_hi
    assert "अज़ाडिरक्टा" not in ans_hi

    ans_kn = rag.query("ಬಿಲ್ವದ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು ಏನು?", use_llm=False)
    assert "Aegle marmelos" in ans_kn
    assert "ಏಗ್ಲೆ" not in ans_kn


def test_short_structured_gemma_response():
    rag = get_service()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "The botanical name of Nimba is Azadirachta indica A. JUSS. (family Meliaceae)."
    }
    with patch("requests.post", return_value=mock_resp) as mock_post:
        ans = rag.query("What is the botanical name of Nimba?", use_llm=True)
        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        payload = call_kwargs.get("json", {})
        options = payload.get("options", {})
        assert options.get("num_predict", 220) <= 80
        assert len(ans.split()) < 35


def test_prompt_evidence_size_reduction():
    rag = get_service()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Bilva belongs to the family Rutaceae."
    }
    with patch("requests.post", return_value=mock_resp) as mock_post:
        rag.query("Which family does Bilva belong to?", use_llm=True)
        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        prompt = call_kwargs.get("json", {}).get("prompt", "")
        assert len(prompt.split()) < 60
        assert "Retrieved Context" not in prompt


def test_no_unsupported_facts_added_by_gemma():
    evidence = {
        "botanical_name": "Azadirachta indica",
        "virya": "Sita",
        "diseases": ["Kustha", "Krmi"],
        "intent": "plant_overview",
        "user_query": "Tell me about Nimba"
    }
    cand_unsupported_virya = "Nimba has an Usna virya and cures many diseases."
    is_g, r = validate_groundedness(cand_unsupported_virya, evidence)
    assert not is_g
    assert "Virya contradictory" in r

    cand_clinical_claim = "Take 500mg daily of Nimba extract for 2 weeks."
    is_g2, r2 = validate_groundedness(cand_clinical_claim, evidence)
    assert not is_g2
    assert "unsupported clinical prescription or dosage" in r2


def test_deterministic_fallback_on_timeout():
    import requests
    rag = get_service()
    init_fallbacks = rag.telemetry.get("deterministic_fallbacks", 0)
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Ollama request timed out")):
        ans = rag.query("What is the botanical name of Nimba?", use_llm=True)
        assert "Azadirachta indica" in ans
        assert "Meliaceae" in ans
        assert rag.telemetry.get("deterministic_fallbacks", 0) > init_fallbacks


# ============================================================
# Task 6 Regression Tests: Universal, Comparison, Multi-Entity Faithfulness
# ============================================================

def test_universal_negative_query():
    rag = get_service()
    # 1. Pattern matching detection
    assert is_universal_claim_query("Does your knowledge base contain evidence that Guduchi treats every disease?")
    assert is_universal_claim_query("Does Nimba treat all diseases?")
    assert is_universal_claim_query("Is Tulsi a cure for everything?")
    assert is_universal_claim_query("Is Ashoka a universal medicine?")

    # 2. Query execution behavior for Guduchi
    ans_guduchi = rag.query("Does your knowledge base contain evidence that Guduchi treats every disease?", use_llm=False)
    assert ans_guduchi.startswith("No, the knowledge base does not support claims that Guduchi treats every disease")
    assert any(term in ans_guduchi for term in ["Prameha", "Jvara", "Kamala", "Kasa", "Svasa"])

    # 3. Query execution behavior for another plant (Nimba)
    ans_nimba = rag.query("Does Nimba treat all diseases?", use_llm=False)
    assert any(phrase in ans_nimba for phrase in ["claims that Nimba treats every disease", "claims that Neem treats every disease"])
    assert any(term in ans_nimba for term in ["Kustha", "Krmi", "Prameha", "Jvara", "Vrana"])


def test_comparison_of_two_unseen_plants():
    rag = get_service()
    ans = rag.query("Compare the Ayurvedic properties of Bilva and Durva.", use_llm=False)
    assert "Property comparison:" in ans
    assert "Bilva" in ans
    assert "Durva" in ans
    assert "Usna" in ans
    assert "Sita" in ans
    assert "Virya" in ans


def test_multi_entity_faithfulness_validation():
    rag = get_service()
    res = rag.resolver.resolve_query("Compare the Ayurvedic properties of Bilva and Durva.")
    full_ctx = extract_full_evidence_context(rag, res, "Compare the Ayurvedic properties of Bilva and Durva.")
    
    # Valid answer referencing actual graph properties for both plants
    valid_ans = "Property comparison: Bilva has Usna Virya (hot potency), whereas Durva has Sita Virya (cool potency)."
    f_score_valid = calculate_faithfulness(valid_ans, valid_ans, full_ctx)
    assert f_score_valid >= 0.95

    # Hallucinated answer claiming unsupported clinical diseases
    hallucinated_ans = "Bilva and Durva cure coronavirus infection, malignant leukemia, and acute pneumonia."
    f_score_hallucinated = calculate_faithfulness(hallucinated_ans, valid_ans, full_ctx)
    assert f_score_hallucinated < 0.60


def test_multi_hop_evidence_union():
    rag = get_service()
    
    # 1. Comparison evidence contains union of Plant A and Plant B properties
    res_comp = rag.resolver.resolve_query("Compare the Ayurvedic properties of Bilva and Durva.")
    ctx_comp = extract_full_evidence_context(rag, res_comp, "Compare the Ayurvedic properties of Bilva and Durva.")
    assert "aegle marmelos" in ctx_comp
    assert "cynodon dactylon" in ctx_comp
    assert "rutaceae" in ctx_comp
    assert "usna" in ctx_comp
    assert "sita" in ctx_comp

    # 2. Disease intersection evidence contains both diseases and common plants
    res_inter = rag.resolver.resolve_query("Which plants are common between Kasa and Svasa?")
    ctx_inter = extract_full_evidence_context(rag, res_inter, "Which plants are common between Kasa and Svasa?")
    assert "kasa" in ctx_inter
    assert "svasa" in ctx_inter
    assert any(plant in ctx_inter for plant in ["justicia adhatoda", "tinospora cordifolia", "abies pindrow", "abies spectabilis"])


def test_concise_conversational_disease_lookup():
    rag = get_service()
    # Conversational paraphrase query
    ans = rag.query("I need a plant connected to Prameha; what does your knowledge base contain?", use_llm=False)
    # Must start directly with factual plant statement, no conversational fluff
    assert not ans.startswith("Certainly!")
    assert not ans.startswith("Sure!")
    assert not ans.startswith("Hello!")
    assert not ans.startswith("According to the Ayurvedic knowledge base")
    assert "Prameha" in ans or "prameha" in ans.lower()
    assert any(term in ans.lower() for term in ["acacia", "tinospora", "guduchi", "plants associated", "medicinal plants"])


def test_no_unsupported_facts_in_comparison_answers():
    evidence = {
        "plant_1": "Bilva (Aegle marmelos)",
        "plant_2": "Durva (Cynodon dactylon)",
        "summary_facts": "Property comparison: Bilva has Usna (hot potency) Virya, whereas Durva has Sita (cool potency) Virya.",
        "intent": "plant_comparison",
        "user_query": "Compare Bilva and Durva"
    }
    # False claim that Bilva is Sita (when Bilva has Usna virya in evidence)
    bad_cand = "Bilva has Sita cold potency and Durva has Usna hot potency."
    is_g, reason = validate_groundedness(bad_cand, evidence)
    assert not is_g
    assert "Virya contradictory" in reason


