import time
import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

prompt_with_vector = """User Question: What is the botanical name of Nimba?
Language: English

Verified Evidence:
- Plant / Sanskrit Name: Nimba
- Botanical Name: Azadirachta indica A. JUSS.
- Botanical Family: Meliaceae
- Rasa (Taste): Tikta, Kashaya
- Virya (Potency): Sita
- Vipaka (Post-digestive effect): Katu
- Documented Indicative Conditions in Graph: Kustha, Prameha, Krmi, Jvara, Vrana, Netraroga, Arsa
- Verified Synonyms: Neem, Nimba, Melia azadirachta L., Antelaea azadirachta (L.) ADELB.
- Retrieved Context Chunks: Azadirachta indica commonly known as neem, nimtree or Indian lilac, is a tree in the mahogany family Meliaceae. It is one of two species in the genus Azadirachta, and is native to the Indian subcontinent and most of the countries in Africa. It is typically grown in tropical and semi-tropical regions. Neem trees also grow on islands in southern Iran. Its fruits and seeds are the source of neem oil. Neem leaves are dried in India and placed in cupboards to prevent insects eating the clothes, and also in tins where rice is stored. Neem is a key component in Ayurvedic medicine and is used for a wide range of ailments, including skin disorders, digestive issues, and parasitic infections. It has antifungal, antibacterial, and antiviral properties. In folk medicine, it is used for fever and malaria.

Answer:"""

sys_p = """You are the multilingual response-generation layer for an Ayurvedic medicinal-plant Hybrid Graph-RAG system.
Answer only from the supplied evidence. Keep botanical Latin names exactly as provided."""

payload = {
    "model": MODEL,
    "system": sys_p,
    "prompt": prompt_with_vector,
    "stream": False,
    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 220},
    "keep_alive": "15m"
}

t0 = time.time()
resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
elapsed = time.time() - t0
d = resp.json()
print("Elapsed wall:", elapsed)
print("Prompt tokens:", d.get("prompt_eval_count"))
print("Prompt eval time (s):", (d.get("prompt_eval_duration", 0) / 1e9))
print("Output tokens:", d.get("eval_count"))
print("Output eval time (s):", (d.get("eval_duration", 0) / 1e9))
print("Response:", d.get("response"))
