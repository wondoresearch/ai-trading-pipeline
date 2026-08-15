import unittest
from datetime import date
from app.final_opportunity.backtest.forward_labels import PricePoint, make_label, audit_labels

class TestForwardLabels(unittest.TestCase):
    def setUp(self):
        self.points=[PricePoint('BBRI',date(2026,8,d),p) for d,p in [(1,100),(2,101),(3,99),(4,105),(5,110),(6,108),(7,115)]]
        self.obs={'ticker':'BBRI','as_of':'2026-08-02','price':101}
    def test_horizon_1_is_next_trading_day(self):
        r=make_label(self.obs,self.points,1)
        self.assertEqual(r.future_day,date(2026,8,3)); self.assertAlmostEqual(r.forward_return,99/101-1)
    def test_horizon_3(self):
        r=make_label(self.obs,self.points,3); self.assertEqual(r.future_day,date(2026,8,5)); self.assertAlmostEqual(r.future_price,110)
    def test_no_lookahead_same_day(self):
        r=make_label({'ticker':'BBRI','as_of':'2026-08-05','price':110},self.points,1)
        self.assertEqual(r.future_day,date(2026,8,6))
    def test_insufficient_future(self):
        r=make_label({'ticker':'BBRI','as_of':'2026-08-06','price':108},self.points,3)
        self.assertFalse(r.eligible); self.assertEqual(r.reason,'insufficient_future_prices')
    def test_audit_rejects_duplicate(self):
        r=make_label(self.obs,self.points,1).to_dict(); a=audit_labels([r,r]); self.assertEqual(a['status'],'FAIL')
    def test_audit_rejects_future_violation(self):
        r=make_label(self.obs,self.points,1).to_dict(); r['future_day']=r['as_of']; a=audit_labels([r]); self.assertEqual(a['status'],'FAIL')

if __name__=='__main__': unittest.main()
