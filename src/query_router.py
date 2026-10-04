import re

class QueryIntentRouter:
    INTENT_PATTERNS = [
        # Check specific Ayurvedic / Sanskrit lookup on a botanical plant first
        ("ayurvedic_identity", [
            r"ayurvedic drug name", r"ayurvedic name", r"ayurvedic identity",
            r"sanskrit name", r"called in ayurveda", r"sanskrit shloka"
        ]),
        # Botanical mapping from drug/common name
        ("botanical_mapping", [
            r"which botanical", r"botanical plant", r"botanical name",
            r"scientific name", r"botanical species", r"corresponds to",
            r"known as .+ in ayurveda", r"associated with"
        ]),
        # Family lookup
        ("family_lookup", [
            r"which family", r"what family", r"family of", r"belong to"
        ]),
        # Vernacular / Local name lookup
        ("vernacular_lookup", [
            r"vernacular", r"local name", r"called in", r"kannada name",
            r"tamil name", r"hindi name"
        ]),
        # Medicinal uses & therapeutics
        ("medicinal_uses", [
            r"medicinal", r"therapeutic", r"cure", r"alleviate",
            r"treatment", r"benefits", r"properties", r"uses"
        ])
    ]

    @classmethod
    def classify_intent(cls, query: str) -> str:
        q_clean = query.lower()
        for intent, patterns in cls.INTENT_PATTERNS:
            if any(re.search(pat, q_clean) for pat in patterns):
                return intent
        return "general"