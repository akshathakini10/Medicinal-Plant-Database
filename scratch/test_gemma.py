import requests
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

url = "http://localhost:11434/api/generate"
sys_prompt = """You are the multilingual response-generation layer for an Ayurvedic medicinal-plant Hybrid Graph-RAG system.
Answer only from the supplied evidence.
Do not use your pretrained knowledge to add unsupported medicinal claims.
Do not invent medicinal uses, diseases, botanical information, Rasa, Virya, Vipaka, dosage, treatment recommendations, or clinical advice.
If the requested information is not available in the evidence, state that it is not available in the current knowledge base.
Answer in the same language as the user's query unless explicitly requested otherwise.
Keep botanical Latin names exactly as provided."""

cases = [
    (
        "What is the botanical name and family of Nimba?",
        "Botanical name: Azadirachta indica A. JUSS.\nFamily: Meliaceae\nCommon name: Nimba / Neem",
    ),
    (
        "नीम का वानस्पतिक नाम क्या है?",
        "Botanical name: Azadirachta indica A. JUSS.\nFamily: Meliaceae\nCommon name: Nimba / Neem (नीम)",
    ),
    (
        "ಬಿಲ್ವದ ಸಸ್ಯಶಾಸ್ತ್ರೀಯ ಹೆಸರು ಏನು?",
        "Botanical name: Aegle marmelos (L.) CORREA E. SCHULTZ\nFamily: Rutaceae\nCommon name: Bilva (ಬಿಲ್ವ)",
    ),
]

for q, ev in cases:
    prompt = f"User Question: {q}\nVerified Evidence:\n{ev}\nAnswer:"
    try:
        resp = requests.post(
            url,
            json={
                "model": "gemma3:12b",
                "system": sys_prompt,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 120},
            },
            timeout=45,
        )
        data = resp.json()
        print(f"Q: {q}")
        print(f"A: {data.get('response', '').strip()}")
        print("-" * 50)
    except Exception as e:
        print(f"Error for {q}: {e}")
