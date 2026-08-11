from datetime import timedelta

from app.news_parser import (
    load_news,
    extract_news_events
)

from app.price_provider import (
    PriceProvider
)

from app.impact_calculator import (
    calculate_impact
)


class ImpactPipeline:

    def __init__(self):

        self.price_provider = (
            PriceProvider()
        )

    def run(
        self,
        news_file
    ):

        news = load_news(
            news_file
        )

        events = extract_news_events(
            news
        )

        results = []

        for event in events:

            published = (
                event["published_at"]
            )

            if not published:
                continue

            # Ambil window cukup lebar
            published_dt = (
                __import__(
                    "dateutil.parser",
                    fromlist=["parse"]
                ).parse(published)
            )

            start = (
                published_dt
                - timedelta(days=2)
            ).date()

            end = (
                published_dt
                + timedelta(days=30)
            ).date()

            prices = (
                self.price_provider
                .get_history(
                    event["ticker"],
                    start,
                    end
                )
            )

            impact = calculate_impact(
                event,
                prices
            )

            if impact:

                results.append(
                    impact
                )

        return results