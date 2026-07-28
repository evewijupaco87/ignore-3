import unittest
from datetime import datetime, timezone

from poker44.validator.runtime_provider import _current_competition_epoch


class CompetitionEpochCadenceTests(unittest.TestCase):
    def test_uses_current_day_epoch_after_20_utc(self):
        epoch = _current_competition_epoch(datetime(2026, 4, 27, 21, 15, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-04-27_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-04-27T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-04-28T20:00:00+00:00")

    def test_uses_previous_day_epoch_before_20_utc(self):
        epoch = _current_competition_epoch(datetime(2026, 4, 27, 19, 59, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-04-26_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-04-26T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-04-27T20:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
