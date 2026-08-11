from app.collector import collect_news
from app.deduplicator import deduplicate
from app.entity_extractor import EntityExtractor
from app.sentiment import SentimentAnalyzer
from app.ticker_mapper import map_tickers


class NewsPipeline:
    def __init__(self, ticker_master_file, sentiment_model=None):
        self.entity_extractor = EntityExtractor(ticker_master_file)
        self.sentiment = (
            SentimentAnalyzer(model_name=sentiment_model)
            if sentiment_model
            else SentimentAnalyzer()
        )

    def run(self, feed_urls):
        articles = collect_news(feed_urls)
        articles = deduplicate(articles)
        results = []

        for article in articles:
            text = " ".join(
                part
                for part in [article.get("title"), article.get("summary")]
                if part
            )

            entities = self.entity_extractor.extract(text)
            if not entities:
                continue

            sentiment = self.sentiment.analyze(text)
            article["entities"] = entities
            article["sentiment"] = sentiment
            article["ticker_mapping"] = map_tickers(article, entities)
            results.append(article)

        return results
