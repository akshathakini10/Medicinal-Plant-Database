import json
import os
import re
from typing import Dict, List, Optional, Any

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

STOP_WORDS = {
    "what", "which", "is", "the", "of", "are", "does", "belong", "to", "in",
    "a", "an", "and", "or", "for", "from", "with", "by", "as", "at", "about",
    "can", "you", "tell", "me", "actually", "known", "called", "name", "names",
    "plant", "plants", "species", "family", "drug", "ayurveda", "ayurvedic",
    "medicinal", "uses", "use", "therapeutic", "benefit", "benefits", "health",
    "cure", "treatment", "associated", "corresponds", "give", "show", "find"
}

KNOWN_ALIASES = {
    "agaru": ["aguru", "agaru", "agar", "aquilaria malaccensis", "aquilaria agallocha"],
    "aguru": ["agaru", "aguru", "agar", "aquilaria malaccensis"],
    "agar": ["agaru", "aguru", "aquilaria malaccensis"],
    "citraka": ["chitraka", "citraka", "plumbago zeylanica"],
    "chitraka": ["citraka", "chitraka", "plumbago zeylanica"],
    "haritaki": ["harad", "haritaki", "terminalia chebula"],
    "harad": ["haritaki", "harad", "terminalia chebula"],
    "guduchi": ["amrita", "giloy", "guduchi", "tinospora cordifolia"],
    "amrita": ["guduchi", "amrita", "giloy"]
}

class PlantEntityResolver:
    def __init__(
        self,
        knowledge_base_path: str = "data/integrated/medicinal_knowledge_base.json",
        ayurvedic_identity_path: str = "data/cleaned/ayurvedic_identity_clean.json",
        shlokas_path: str = "data/cleaned/shlokas_clean.json"
    ):
        self.plants_by_id: Dict[str, Dict[str, Any]] = {}
        self.botanical_index: Dict[str, List[str]] = {}
        self.ayurvedic_index: Dict[str, List[str]] = {}
        self.vernacular_index: Dict[str, List[str]] = {}
        self.shlokas_by_drug: Dict[str, List[Dict[str, Any]]] = {}
        self.botanical_to_id: Dict[str, str] = {}

        self._load_data(knowledge_base_path, ayurvedic_identity_path, shlokas_path)

    def _index_term(self, index: Dict[str, List[str]], term: str, plant_id: str):
        norm = normalize_text(term)
        if not norm or len(norm) < 2 or not plant_id:
            return
        if plant_id not in index.setdefault(norm, []):
            index[norm].append(plant_id)

        if norm in KNOWN_ALIASES:
            for alias in KNOWN_ALIASES[norm]:
                if plant_id not in index.setdefault(alias, []):
                    index[alias].append(plant_id)

    def _load_data(self, kb_path: str, ayur_path: str, shlokas_path: str):
        # 1. Integrated Knowledge Base
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)

            for record in kb_data:
                pid = record.get("plant_id")
                if not pid:
                    continue
                self.plants_by_id[pid] = record

                b_name = record.get("botanical_name", "")
                norm_bname = normalize_text(b_name)
                if norm_bname:
                    self.botanical_to_id[norm_bname] = pid
                    self._index_term(self.botanical_index, b_name, pid)
                    tokens = norm_bname.split()
                    if len(tokens) >= 2:
                        binomial = " ".join(tokens[:2])
                        self.botanical_to_id[binomial] = pid
                        self._index_term(self.botanical_index, binomial, pid)

                for syn in record.get("synonyms") or []:
                    self._index_term(self.botanical_index, str(syn), pid)
                    self.botanical_to_id[normalize_text(str(syn))] = pid

                v_names = record.get("vernacular_names")
                if isinstance(v_names, str):
                    cleaned_v = re.sub(r'[A-Za-z]+\s*\(\d+\)\s*-', ',', v_names)
                    for item in cleaned_v.split(","):
                        clean_item = normalize_text(item)
                        if clean_item and len(clean_item) > 2 and clean_item not in STOP_WORDS:
                            self._index_term(self.vernacular_index, clean_item, pid)

                for item in record.get("ayurvedic_identity") or []:
                    if isinstance(item, dict):
                        d = item.get("drug_name") or item.get("drug") or ""
                        t = item.get("transliteration") or ""
                        self._index_term(self.ayurvedic_index, d, pid)
                        self._index_term(self.ayurvedic_index, t, pid)

        # 2. Standalone Ayurvedic Identity
        if os.path.exists(ayur_path):
            with open(ayur_path, "r", encoding="utf-8") as f:
                ayur_data = json.load(f)

            for item in ayur_data:
                pid = item.get("plant_id")
                if not pid or pid not in self.plants_by_id:
                    bot_corr = normalize_text(item.get("botanical_correlation") or item.get("botanical_name") or "")
                    pid = self.botanical_to_id.get(bot_corr)
                    if not pid and bot_corr:
                        tokens = bot_corr.split()
                        if len(tokens) >= 2:
                            pid = self.botanical_to_id.get(" ".join(tokens[:2]))

                if pid:
                    drug = item.get("drug_name") or item.get("drug") or ""
                    translit = item.get("transliteration") or ""
                    self._index_term(self.ayurvedic_index, drug, pid)
                    self._index_term(self.ayurvedic_index, translit, pid)

        # 3. Shlokas Dataset
        if os.path.exists(shlokas_path):
            with open(shlokas_path, "r", encoding="utf-8") as f:
                shloka_data = json.load(f)

            for s_entry in shloka_data:
                drug = normalize_text(s_entry.get("drug_name") or "")
                translit = normalize_text(s_entry.get("transliteration") or "")
                pid = s_entry.get("plant_id")

                if not pid or pid not in self.plants_by_id:
                    bot_corr = normalize_text(s_entry.get("botanical_name") or s_entry.get("botanical_correlation") or "")
                    pid = self.botanical_to_id.get(bot_corr)
                    if not pid and bot_corr:
                        tokens = bot_corr.split()
                        if len(tokens) >= 2:
                            pid = self.botanical_to_id.get(" ".join(tokens[:2]))

                for k in [drug, translit]:
                    if k:
                        self.shlokas_by_drug.setdefault(k, []).append(s_entry)
                        if k in KNOWN_ALIASES:
                            for alias in KNOWN_ALIASES[k]:
                                self.shlokas_by_drug.setdefault(alias, []).append(s_entry)
                        if pid:
                            self._index_term(self.ayurvedic_index, k, pid)

        print(f"Indexed {len(self.plants_by_id)} plants, {len(self.botanical_index)} botanical names, and {len(self.ayurvedic_index)} Ayurvedic terms.")

    def resolve_entity(self, query: str) -> Optional[Dict[str, Any]]:
        norm_query = normalize_text(query)
        words = norm_query.split()
        max_n = min(len(words), 5)

        # 1. Check Ayurvedic drug index
        for n in range(max_n, 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if n == 1 and phrase in STOP_WORDS:
                    continue

                if phrase in self.ayurvedic_index:
                    pids = self.ayurvedic_index[phrase]
                    matched_records = [self.plants_by_id[p] for p in pids if p in self.plants_by_id]
                    if matched_records:
                        return {
                            "matched_term": phrase,
                            "entity_type": "ayurvedic_drug",
                            "plant_ids": pids,
                            "records": matched_records,
                            "shlokas": self.shlokas_by_drug.get(phrase, [])
                        }

        # 2. Check Botanical Name index
        for n in range(max_n, 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if n == 1 and phrase in STOP_WORDS:
                    continue

                if phrase in self.botanical_index:
                    pids = self.botanical_index[phrase]
                    matched_records = [self.plants_by_id[p] for p in pids if p in self.plants_by_id]
                    if matched_records:
                        shlokas = []
                        for r in matched_records:
                            for s in r.get("sanskrit_shlokas", []):
                                if s and s not in shlokas:
                                    shlokas.append(s)
                            for ayur in r.get("ayurvedic_identity", []):
                                d_name = normalize_text(ayur.get("drug_name", ""))
                                for s in self.shlokas_by_drug.get(d_name, []):
                                    if s not in shlokas:
                                        shlokas.append(s)

                        return {
                            "matched_term": phrase,
                            "entity_type": "botanical_name",
                            "plant_ids": pids,
                            "records": matched_records,
                            "shlokas": shlokas
                        }

        # 3. Check Vernacular index
        for n in range(max_n, 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if n == 1 and phrase in STOP_WORDS:
                    continue

                if phrase in self.vernacular_index:
                    pids = self.vernacular_index[phrase]
                    matched_records = [self.plants_by_id[p] for p in pids if p in self.plants_by_id]
                    if matched_records:
                        return {
                            "matched_term": phrase,
                            "entity_type": "vernacular_name",
                            "plant_ids": pids,
                            "records": matched_records,
                            "shlokas": self.shlokas_by_drug.get(phrase, [])
                        }

        return None