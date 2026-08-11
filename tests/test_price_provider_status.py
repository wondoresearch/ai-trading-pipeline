import unittest
from unittest.mock import patch

import pandas as pd

from app.price_provider import PriceProvider


class TestPriceProviderStatus(unittest.TestCase):
    def test_error_payload_is_structured(self):
        provider = PriceProvider()
        payload = provider._error_payload("yahoo_chart", RuntimeError("timeout"))
        self.assertEqual(
            payload,
            {
                "stage": "yahoo_chart",
                "error_type": "RuntimeError",
                "message": "timeout",
            },
        )

    def test_get_history_with_status_catches_unexpected_provider_error(self):
        provider = PriceProvider()
        with patch.object(provider, "get_history", side_effect=RuntimeError("network down")):
            history, status = provider.get_history_with_status("BBCA", "2026-08-10", "2026-08-12")

        self.assertTrue(history.empty)
        self.assertEqual(status, "provider_error")
        self.assertEqual(provider.last_error["stage"], "get_history")
        self.assertEqual(provider.last_error["error_type"], "RuntimeError")
        self.assertEqual(provider.last_error["message"], "network down")

    def test_get_history_with_status_preserves_provider_error(self):
        provider = PriceProvider()
        provider.last_error = {
            "stage": "yahoo_chart",
            "error_type": "RequestException",
            "message": "timeout",
        }
        with patch.object(provider, "get_history", return_value=pd.DataFrame()):
            history, status = provider.get_history_with_status("BBCA", "2026-08-10", "2026-08-12")

        self.assertTrue(history.empty)
        self.assertEqual(status, "provider_error")
        self.assertEqual(provider.last_error["stage"], "yahoo_chart")

    def test_get_history_with_status_returns_ok_for_data(self):
        provider = PriceProvider()
        frame = pd.DataFrame({"Date": pd.to_datetime(["2026-08-10"]), "Close": [100.0]})
        with patch.object(provider, "get_history", return_value=frame):
            history, status = provider.get_history_with_status("BBCA", "2026-08-10", "2026-08-12")

        self.assertEqual(status, "ok")
        self.assertFalse(history.empty)
        self.assertIsNone(provider.last_error)


if __name__ == "__main__":
    unittest.main()
