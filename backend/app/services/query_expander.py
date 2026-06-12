import re
from typing import Dict

class MedicalQueryExpander:
    # Common medical abbreviations mapped to their full clinical terms
    ACRONYMS: Dict[str, str] = {
        "COPD": "chronic obstructive pulmonary disease",
        "HF": "heart failure",
        "HFPEF": "heart failure with preserved ejection fraction",
        "HFREF": "heart failure with reduced ejection fraction",
        "CAD": "coronary artery disease",
        "DM": "diabetes mellitus",
        "HTN": "hypertension",
        "ACEI": "angiotensin converting enzyme inhibitor",
        "ARB": "angiotensin receptor blocker",
        "BB": "beta blocker",
        "CKD": "chronic kidney disease",
        "MI": "myocardial infarction",
        "AF": "atrial fibrillation",
        "AFIB": "atrial fibrillation"
    }

    @classmethod
    def expand_query(cls, query: str) -> str:
        """
        Expands abbreviations in a search query.
        Returns the expanded query, keeping original terms alongside expansions.
        Example: "COPD treatment" -> "COPD (chronic obstructive pulmonary disease) treatment"
        """
        if not query:
            return ""

        expanded_terms = []
        # Split by words but preserve punctuation spaces
        words = re.findall(r'\b\w+\b|\s+|[^\w\s]', query)

        for token in words:
            # Check if token matches clinical acronym list (case insensitive)
            cleaned_token = token.upper().strip()
            if cleaned_token in cls.ACRONYMS:
                expansion = cls.ACRONYMS[cleaned_token]
                # Append both the abbreviation and its full definition
                expanded_terms.append(f"{token} ({expansion})")
            else:
                expanded_terms.append(token)

        return "".join(expanded_terms)
