from pathlib import Path

test_file = Path("tests/test_regression.py")
content = test_file.read_text(encoding="utf-8")

new_tests = """

# ============================================================
# Additional Comprehensive Regression Tests (Part 10 Requirements)
# ============================================================

def test_hindi_complex_property_answer_quality():
    rag = get_service()
    ans = rag.query("गुडूची का वानस्पतिक नाम और वीर्य बताइए।", use_llm=False)
    assert re.search(r"[\\u0900-\\u097F]", ans) is not None
    assert "Tinospora cordifolia" in ans
    assert any(term in ans for term in ["उष्ण", "Usna", "वीर्य"])


def test_kannada_complex_property_answer_quality():
    rag = get_service()
    ans = rag.query("ಗುಡುಚಿಯ ಆಯುರ್ವೇದ ಗುಣಗಳನ್ನು ತಿಳಿಸಿ.", use_llm=False)
    assert re.search(r"[\\u0C80-\\u0CFF]", ans) is not None
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
"""

test_file.write_text(content.strip() + "\n" + new_tests, encoding="utf-8")
print("Appended tests successfully.")
