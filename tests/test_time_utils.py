import unittest

from app.time_utils import normalize_timezone, normalize_utc, parse_datetime


class TestTimeUtils(unittest.TestCase):
    def test_parse_rfc822_timestamp(self):
        dt = parse_datetime("Mon, 10 Aug 2026 09:30:00 +0700")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.utcoffset().total_seconds(), 7 * 3600)

    def test_normalize_utc(self):
        self.assertEqual(
            normalize_utc("2026-08-10T09:30:00+07:00"),
            "2026-08-10T02:30:00+00:00",
        )

    def test_normalize_jakarta(self):
        self.assertEqual(
            normalize_timezone("2026-08-10T02:30:00+00:00"),
            "2026-08-10T09:30:00+07:00",
        )

    def test_invalid_timestamp(self):
        self.assertIsNone(normalize_utc("not-a-date"))


if __name__ == "__main__":
    unittest.main()
