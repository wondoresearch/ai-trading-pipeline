import unittest

from app.news_parser import extract_news_events


class TestNewsParser(unittest.TestCase):
    def test_article_becomes_one_event_per_ticker(self):
        news = [
            {
                "id": "news-1",
                "source": "Test Feed",
                "title": "BBRI and BMRI report strong results",
                "summary": "Example",
                "url": "https://example.com/news-1",
                "published_at": "2026-08-10T09:30:00+07:00",
                "sentiment": {
                    "label": "positive",
                    "score": 0.91,
                    "signed_score": 0.91,
                },
                "ticker_mapping": [
                    {
                        "ticker": "BBRI",
                        "company": "Bank Rakyat Indonesia",
                        "matched_alias": "BBRI",
                        "entity_confidence": 1.0,
                    },
                    {
                        "ticker": "BMRI",
                        "company": "Bank Mandiri",
                        "matched_alias": "BMRI",
                        "entity_confidence": 1.0,
                    },
                ],
            }
        ]

        events = extract_news_events(news)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["ticker"], "BBRI")
        self.assertEqual(events[1]["ticker"], "BMRI")
        self.assertEqual(
            events[0]["published_at_utc"],
            "2026-08-10T02:30:00+00:00",
        )
        self.assertEqual(
            events[0]["published_at"],
            "2026-08-10T09:30:00+07:00",
        )
        self.assertEqual(events[0]["schema_version"], "1.0")
        self.assertTrue(events[0]["event_id"])

    def test_empty_ticker_is_skipped(self):
        news = [
            {
                "id": "news-2",
                "source": "Test Feed",
                "title": "Example",
                "summary": "",
                "url": "https://example.com/news-2",
                "published_at": "2026-08-10T09:30:00+07:00",
                "sentiment": {},
                "ticker_mapping": [{"ticker": ""}],
            }
        ]
        self.assertEqual(extract_news_events(news), [])


if __name__ == "__main__":
    unittest.main()
