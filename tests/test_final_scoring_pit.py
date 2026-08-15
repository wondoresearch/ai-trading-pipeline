import unittest
from app.final_opportunity.scoring import score_ticker

class TestFinalScoringPIT(unittest.TestCase):
    def prices(self):
        return [
            {"trading_date":"2026-08-10","close":100,"adj_close":100},
            {"trading_date":"2026-08-11","close":102,"adj_close":102},
            {"trading_date":"2026-08-12","close":104,"adj_close":104},
            {"trading_date":"2026-08-15","close":150,"adj_close":150},
        ]

    def test_future_price_is_not_used_when_cutoff_is_applied(self):
        all_rows=self.prices()
        safe=[x for x in all_rows if x["trading_date"] <= "2026-08-12"]
        r=score_ticker("BBRI",safe,[],1,1,0)
        self.assertAlmostEqual(r.momentum,104/102-1.0)

    def test_adjusted_close_is_return_basis(self):
        prices=[
            {"trading_date":"2026-08-10","close":100,"adj_close":80},
            {"trading_date":"2026-08-11","close":110,"adj_close":88},
        ]
        r=score_ticker("BBRI",prices,[],1,1,0)
        self.assertAlmostEqual(r.momentum,.10)

if __name__=="__main__": unittest.main()
