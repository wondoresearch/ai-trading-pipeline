from difflib import SequenceMatcher

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def deduplicate(articles, threshold=0.88):
    unique = []
    seen_ids = set()
    for article in articles:
        if article["id"] in seen_ids:
            continue
        duplicate = any(
            similarity(article["title"], existing["title"]) >= threshold
            for existing in unique
        )
        if not duplicate:
            unique.append(article)
            seen_ids.add(article["id"])
    return unique
