import csv, json, tempfile
from pathlib import Path
import unittest
import subprocess, sys


class TestPITBundle(unittest.TestCase):
    def test_builder_rejects_future_publication(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            inp = p / "in.json"
            out = p / "out.json"
            inp.write_text(json.dumps([{
                "ticker":"BBRI","as_of":"2026-03-31","publication_date":"2026-04-01",
                "price":4200,"fundamental_score":0.8,"eligible":True
            }]))
            r = subprocess.run([sys.executable, "scripts/build_pit_dataset.py", str(inp), "-o", str(out)])
            self.assertEqual(r.returncode, 0)
            self.assertEqual(json.loads(out.read_text()), [])

    def test_observation_adapter_does_not_invent_news(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            m = p / "BBRI.csv"
            f = p / "BBRI_annual.json"
            o = p / "obs.json"
            with m.open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=["Date","Close"])
                w.writeheader(); w.writerow({"Date":"2026-03-15","Close":"4200"})
            f.write_text(json.dumps([{
                "ticker":"BBRI","period_end":"2025-12-31",
                "publication_date":"2026-03-15","revenue":100,"net_income":20
            }]))
            r = subprocess.run([
                sys.executable,"scripts/build_pit_observations.py",
                "--tickers","BBRI","--market-dir",str(p),
                "--financial-dir",str(p),"-o",str(o)
            ])
            self.assertEqual(r.returncode, 0)
            rows = json.loads(o.read_text())
            self.assertEqual(len(rows),1)
            self.assertIsNone(rows[0]["news_score"])
            self.assertTrue(rows[0]["eligible"])

    def test_audit_passes_valid_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); inp = p/"x.json"
            inp.write_text(json.dumps([{
                "ticker":"PTBA","as_of":"2026-03-31",
                "publication_date":"2026-03-30","price":3000,
                "eligible":True
            }]))
            r = subprocess.run([sys.executable,"scripts/audit_pit_dataset.py",str(inp)])
            self.assertEqual(r.returncode,0)


if __name__ == "__main__":
    unittest.main()
