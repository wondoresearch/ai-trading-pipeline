import json
import re

class EntityExtractor:
    def __init__(self, master_file):
        with open(master_file, "r", encoding="utf-8") as f:
            self.companies = json.load(f)

    def extract(self, text: str):
        found = []
        text_lower = text.lower()
        for company in self.companies:
            for alias in company["aliases"]:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, text_lower):
                    found.append({
                        "ticker": company["ticker"],
                        "company": company["name"],
                        "matched_alias": alias,
                        "confidence": 1.0
                    })
                    break
        return found
