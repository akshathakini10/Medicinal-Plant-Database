import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Centralized Model Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "240"))
