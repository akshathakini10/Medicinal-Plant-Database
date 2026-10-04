import time
import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

def run_profile_query(prompt, system_prompt, options=None, keep_alive="10m"):
    payload = {
        "model": MODEL,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": options or {"temperature": 0.1, "top_p": 0.9, "num_predict": 100},
        "keep_alive": keep_alive
    }
    
    t_start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
        t_end = time.time()
        elapsed = t_end - t_start
        if resp.status_code == 200:
            data = resp.json()
            # Ollama provides detailed metrics:
            # total_duration, load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration
            return {
                "success": True,
                "elapsed_wall": elapsed,
                "response": data.get("response", ""),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "load_duration_ms": (data.get("load_duration", 0) / 1e6),
                "prompt_eval_duration_ms": (data.get("prompt_eval_duration", 0) / 1e6),
                "eval_duration_ms": (data.get("eval_duration", 0) / 1e6),
                "total_duration_ms": (data.get("total_duration", 0) / 1e6),
            }
        else:
            return {"success": False, "elapsed_wall": elapsed, "error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"success": False, "elapsed_wall": time.time() - t_start, "error": str(e)}

if __name__ == "__main__":
    print("Testing Call 1 (Possible Cold Start / Warmup)...")
    p1 = "User Question: What is the botanical name of Nimba?\nLanguage: English\n\nVerified Evidence:\n- Plant: Nimba\n- Botanical Name: Azadirachta indica A. JUSS.\n- Family: Meliaceae\n\nAnswer:"
    sys_p = "You are the multilingual response-generation layer for an Ayurvedic medicinal-plant Hybrid Graph-RAG system. Answer concisely in one sentence using only verified evidence."
    
    res1 = run_profile_query(p1, sys_p)
    print("Call 1 result:", json.dumps(res1, indent=2, ensure_ascii=False))

    print("\nTesting Call 2 (Warm Query)...")
    p2 = "User Question: What is the Virya of Bilva?\nLanguage: English\n\nVerified Evidence:\n- Plant: Bilva\n- Virya: Usna (warm/hot)\n\nAnswer:"
    res2 = run_profile_query(p2, sys_p)
    print("Call 2 result:", json.dumps(res2, indent=2, ensure_ascii=False))
