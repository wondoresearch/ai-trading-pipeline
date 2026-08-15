import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from publication_evidence import parse_publication_evidence
class TestPublicationEvidenceV1(unittest.TestCase):
 def setUp(self): self.h=(Path(__file__).parent/"fixture.html").read_text()
 def test_four_tickers(self):
  r=parse_publication_evidence(self.h,["BBRI","BBCA","PTBA","ADRO"]); self.assertEqual(len(r),4)
 def test_timestamp(self):
  r=parse_publication_evidence(self.h,["BBRI"]); self.assertEqual(r[0]["publication_timestamp"],"2021-02-09T18:34:55")
 def test_filter(self):
  r=parse_publication_evidence(self.h,["PTBA"]); self.assertEqual(r[0]["ticker"],"PTBA")
 def test_period_not_publication(self):
  r=parse_publication_evidence(self.h,["ADRO"]); self.assertEqual(r[0]["financial_year"],2020); self.assertEqual(r[0]["publication_date"],"2021-03-04")
if __name__=="__main__": unittest.main()
