def map_tickers(article, entities):
    mappings = []
    for entity in entities:
        mappings.append({
            "ticker": entity["ticker"],
            "company": entity["company"],
            "matched_alias": entity["matched_alias"],
            "entity_confidence": entity["confidence"],
            "sentiment": article["sentiment"]
        })
    return mappings
