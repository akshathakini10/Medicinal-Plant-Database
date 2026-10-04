import time
import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

SYSTEM_PROMPT = """You are the multilingual response-generation layer for an Ayurvedic medicinal-plant Hybrid Graph-RAG system.
Your task is to synthesize the verified evidence into a precise, natural answer that directly answers the user's question.

STRICT RULES:
1. Answer in the exact language and style of the user query:
   - English query -> Answer in concise English.
   - Hindi query (Devanagari) -> Answer in natural Hindi using Devanagari script.
   - Kannada query (Kannada script) -> Answer in natural Kannada using Kannada script.
   - Hindi-English code-mixed query (e.g. "Tulsi kis family ko belong karti hai?") -> Answer in natural Hindi-English code-mix (e.g. "Tulsi (Ocimum tenuiflorum) Lamiaceae family ko belong karti hai.").
   - Kannada-English code-mixed query (e.g. "Bilvada virya enu?") -> Answer in natural Kannada-English code-mix (e.g. "Bilvada (Aegle marmelos) Virya Usna (hot potency) aagide.").
2. Botanical Latin names MUST remain in exact Latin script (e.g. 'Azadirachta indica A. JUSS.', 'Aegle marmelos'). Never translate or transliterate botanical names.
3. For botanical and family queries, mention both the botanical name and family (e.g. with '(family ...)', '(कुल: ...)', or '(ಕುಟುಂಬ: ...)').
4. Answer ALL requested fields completely without omitting any.
5. Use ONLY the supplied Locked Facts. Do not add outside diseases, dosage, or clinical advice."""

test_cases = [
    {
        "id": "HOLD-14",
        "q": "नीम का वानस्पतिक नाम क्या है?",
        "lang": "Hindi",
        "evidence": {
            "entity": "Nimba (नीम)",
            "botanical_name": "Azadirachta indica A. JUSS.",
            "family": "Meliaceae"
        },
        "requested_fields": ["botanical_name", "family"],
        "num_predict": 70
    },
    {
        "id": "HOLD-16",
        "q": "ಬಿಲ್ವದ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು ಏನು?",
        "lang": "Kannada",
        "evidence": {
            "entity": "Bilva (ಬಿಲ್ವ)",
            "botanical_name": "Aegle marmelos (L.) CORREA E. SCHULTZ",
            "family": "Rutaceae"
        },
        "requested_fields": ["botanical_name", "family"],
        "num_predict": 70
    },
    {
        "id": "HOLD-17",
        "q": "Bilvada virya enu?",
        "lang": "Kannada+English",
        "evidence": {
            "entity": "Bilva",
            "botanical_name": "Aegle marmelos",
            "virya": "Usna (hot potency)"
        },
        "requested_fields": ["virya"],
        "num_predict": 60
    },
    {
        "id": "HOLD-18",
        "q": "Tulsi kis family ko belong karti hai?",
        "lang": "Hindi+English",
        "evidence": {
            "entity": "Tulsi",
            "botanical_name": "Ocimum tenuiflorum",
            "family": "Lamiaceae"
        },
        "requested_fields": ["family"],
        "num_predict": 60
    },
    {
        "id": "DEV-54",
        "q": "गुडूची का वानस्पतिक नाम और वीर्य बताइए।",
        "lang": "Hindi",
        "evidence": {
            "entity": "Guduchi (गुडूची)",
            "botanical_name": "Tinospora cordifolia (WILLD.) HOOK.F. & THOMS.",
            "virya": "उष्ण (Usna)"
        },
        "requested_fields": ["botanical_name", "virya"],
        "num_predict": 80
    },
    {
        "id": "DEV-55",
        "q": "ಗುಡುಚಿಯ ಆಯುರ್ವೇದ ಗುಣಗಳನ್ನು ತಿಳಿಸಿ.",
        "lang": "Kannada",
        "evidence": {
            "entity": "Guduchi (ಗುಡುಚಿ)",
            "rasa": "ತಿಕ್ತ (Tikta)",
            "virya": "ಉಷ್ಣ (Usna)",
            "vipaka": "ಮಧುರ / ಕಟುಕ (Madhura / Katuka)"
        },
        "requested_fields": ["rasa", "virya", "vipaka"],
        "num_predict": 90
    }
]

for tc in test_cases:
    ev_str = "\n".join(f"- {k}: {v}" for k, v in tc["evidence"].items())
    prompt = (
        f"User Question: {tc['q']}\n"
        f"Language Style: {tc['lang']}\n"
        f"Requested Fields: {', '.join(tc['requested_fields'])}\n\n"
        f"Locked Facts:\n{ev_str}\n\n"
        f"Answer:"
    )
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": tc["num_predict"]},
        "keep_alive": "30m"
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    dur = time.time() - t0
    ans = resp.json().get("response", "").strip()
    print(f"[{tc['id']}] ({dur:.2f}s) Q: {tc['q']}")
    print(f"  -> {ans}\n")
