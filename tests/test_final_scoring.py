import unittest
from app.final_opportunity.scoring import score_ticker

class TestFinalScoring(unittest.TestCase):
    def setUp(self):
        self.prices = [{"close": 100+i*0.2, "trading_date": f"2025-01-{(i%28)+1:02d}"} for i in range(140)]
        self.news = [{"sentiment": 0.5, "title":"positive", "published_at":"2026-08-01"} for _ in range(6)]

    def test_deterministic(self):
        a = score_ticker("BBRI", self.prices, self.news, 20, 120, 5)
        b = score_ticker("BBRI", self.prices, self.news, 20, 120, 5)
        self.assertEqual(a, b)

    def test_insufficient_price(self):
        with self.assertRaises(ValueError):
            score_ticker("BBRI", self.prices[:20], self.news, 20, 120, 5)

    def test_negative_sentiment_does_not_create_upside(self):
        news = [{"sentiment": -1.0, "title":"bad", "published_at":"2026-08-01"} for _ in range(6)]
        r = score_ticker("BBRI", self.prices, news, 20, 120, 5)
        self.assertGreaterEqual(r.score, 0.0)

if __name__ == "__main__":
    unittest.main()
