import unittest

from app.event_schema import validate_event


class TestEventSchema(unittest.TestCase):
    def test_valid_event(self):
        validate_event(
            {
                "event_id": "e1",
                "news_id": "n1",
                "ticker": "BBRI",
                "title": "Test",
                "url": "https://example.com",
                "source": "Test",
                "published_timezone": "Asia/Jakarta",
                "schema_version": "1.0",
                "sentiment_score": 0.8,
                "entity_confidence": 1.0,
            }
        )

    def test_missing_required_field(self):
        with self.assertRaises(ValueError):
            validate_event({"event_id": "e1", "news_id": "n1"})

    def test_invalid_sentiment_score(self):
        with self.assertRaises(ValueError):
            validate_event(
                {
                    "event_id": "e1",
                    "news_id": "n1",
                    "ticker": "BBRI",
                    "title": "Test",
                    "url": "https://example.com",
                    "source": "Test",
                    "published_timezone": "Asia/Jakarta",
                    "schema_version": "1.0",
                    "sentiment_score": 1.5,
                }
            )


if __name__ == "__main__":
    unittest.main()
