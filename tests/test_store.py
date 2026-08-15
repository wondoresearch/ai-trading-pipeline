import tempfile, unittest
from pathlib import Path
from app.final_opportunity.store import Store

class TestStore(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)/"x.db")
            s.upsert_universe([{"ticker":"BBRI","name":"Bank Rakyat Indonesia","exchange":"XIDX","updated_at":"2026-08-12T00:00:00Z"}])
            self.assertEqual(s.universe()[0]["ticker"],"BBRI")
            s.upsert_prices([{"ticker":"BBRI","trading_date":"2026-08-12","close":100,"source":"test","retrieved_at":"2026-08-12T00:00:00Z"}])
            self.assertEqual(s.prices("BBRI")[0]["close"],100)

if __name__ == "__main__":
    unittest.main()
