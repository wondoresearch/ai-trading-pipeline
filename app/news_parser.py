import json


def load_news(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def extract_news_events(news):

    events = []

    for article in news:

        published_at = article.get(
            "published_at"
        )

        mappings = article.get(
            "ticker_mapping",
            []
        )

        sentiment = article.get(
            "sentiment",
            {}
        )

        for mapping in mappings:

            events.append({

                "news_id": article["id"],

                "ticker":
                    mapping["ticker"],

                "title":
                    article["title"],

                "url":
                    article["url"],

                "published_at":
                    published_at,

                "sentiment":
                    sentiment.get(
                        "label"
                    ),

                "sentiment_score":
                    sentiment.get(
                        "score"
                    ),

                "signed_score":
                    sentiment.get(
                        "signed_score",
                        0
                    )
            })

    return events