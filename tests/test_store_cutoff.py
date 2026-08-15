import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from app.final_opportunity.store import Store

class TestStoreCutoff(unittest.TestCase):
    def test_prices_can_be_cut_at_as_of(self):
        with TemporaryDirectory() as td:
            s=Store(Path(td)/"research.db")
            s.upsert_prices([
                {"ticker":"BBRI","trading_date":"2026-08-12","close":100,"adj_close":100,"source":"test","retrieved_at":"2026-08-15T00:00:00Z"},
                {"ticker":"BBRI","trading_date":"2026-08-15","close":150,"adj_close":150,"source":"test","retrieved_at":"2026-08-15T00:00:00Z"},
            ])
            self.assertEqual([r["trading_date"] for r in s.prices("BBRI","2026-08-12")],["2026-08-12"])

if __name__=="__main__": unittest.main()
