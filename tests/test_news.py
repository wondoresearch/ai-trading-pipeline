import unittest
from app.final_opportunity.news import lexicon_sentiment

class TestNews(unittest.TestCase):
    def test_positive(self):
        score,label=lexicon_sentiment("laba tumbuh dan dividen meningkat")
        self.assertGreater(score,0); self.assertEqual(label,"positive")
    def test_negative(self):
        score,label=lexicon_sentiment("rugi dan utang meningkat")
        self.assertLess(score,0); self.assertEqual(label,"negative")
    def test_neutral(self):
        score,label=lexicon_sentiment("rapat perusahaan berlangsung")
        self.assertEqual(score,0); self.assertEqual(label,"neutral")

if __name__ == "__main__":
    unittest.main()
