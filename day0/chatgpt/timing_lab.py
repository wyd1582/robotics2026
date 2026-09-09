"""A synthetic timing exercise, not a production controller or real dataset audit.

All timestamps are seconds in one explicitly shared clock domain and episode.
sample_time: time the physical quantity was observed.
available_time: earliest time that observation can be used by the policy.
Run: python3 timing_lab.py; python3 timing_lab.py --test
No third-party dependencies, network, files, or hardware access.
"""
from dataclasses import dataclass
import math
import sys
import unittest


@dataclass(frozen=True)
class Sample:
    episode: str
    clock: str
    sample_time: float
    available_time: float
    value: float


def select_observation(samples, decision_time, max_age, *, episode, clock):
    """Return freshest causally available sample, or None if no usable sample.

    Reject invalid inputs; never relabel timestamps or cross episode/reset bounds.
    A clock reset must be represented by a new clock identifier.
    """
    if not math.isfinite(decision_time) or not math.isfinite(max_age) or max_age < 0:
        raise ValueError("Invalid decision time or age budget")
    selected = [s for s in samples if s.episode == episode and s.clock == clock]
    previous = -math.inf
    for s in selected:
        if not all(math.isfinite(v) for v in (s.sample_time, s.available_time, s.value)):
            raise ValueError("Non-finite input")
        if s.sample_time <= previous:
            raise ValueError("Sample times must be strictly increasing within each stream")
        if s.available_time < s.sample_time:
            raise ValueError("Availability before sampling: check clocks and units")
        previous = s.sample_time
    usable = [s for s in selected if s.available_time <= decision_time
              and 0 <= decision_time - s.sample_time <= max_age]
    return max(usable, key=lambda s: s.sample_time, default=None)


class TimingTests(unittest.TestCase):
    def sample(self, t, arrival, value=1, episode="e1", clock="c1"):
        return Sample(episode, clock, t, arrival, value)

    def pick(self, samples, t=1.0, age=0.25):
        return select_observation(samples, t, age, episode="e1", clock="c1")

    def test_newest_available(self):
        a, b = self.sample(.8, .85), self.sample(.9, .95)
        self.assertEqual(self.pick([a, b]), b)

    def test_no_future_arrival_leakage(self):
        a, b = self.sample(.8, .85), self.sample(.99, 1.10)
        self.assertEqual(self.pick([a, b]), a)

    def test_stale_is_none(self):
        self.assertIsNone(self.pick([self.sample(.5, .6)]))

    def test_inclusive_age_boundary(self):
        a = self.sample(.75, .8)
        self.assertEqual(self.pick([a]), a)

    def test_episode_isolation(self):
        self.assertIsNone(self.pick([self.sample(.9, .95, episode="e2")]))

    def test_clock_reset_isolation(self):
        self.assertIsNone(self.pick([self.sample(.9, .95, clock="c2")]))

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            self.pick([self.sample(.9, .91), self.sample(.9, .92)])

    def test_reverse_time_rejected(self):
        with self.assertRaises(ValueError):
            self.pick([self.sample(.9, .95), self.sample(.8, .85)])

    def test_impossible_arrival_rejected(self):
        with self.assertRaises(ValueError):
            self.pick([self.sample(.9, .8)])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            self.pick([self.sample(.9, .95, value=float("nan"))])

    def test_empty(self):
        self.assertIsNone(self.pick([]))

    def test_out_of_order_arrivals(self):
        a, b = self.sample(.8, 1.1), self.sample(.9, .95)
        self.assertEqual(self.pick([a, b]), b)


def demo():
    observations = [Sample("e1", "c1", .80, .85, 10.0),
                    Sample("e1", "c1", .99, 1.10, 20.0)]
    naive = min(observations, key=lambda s: abs(s.sample_time - 1.0))
    causal = select_observation(observations, 1.0, .25, episode="e1", clock="c1")
    print("Synthetic decision at t=1.00 seconds")
    print("Nearest sample-time join picks:", naive)
    print("Causal join picks:", causal)
    print("The nearest sample was not available until t=1.10: it leaks future information.")
    print("Exercise: sweep max_age; count unavailable decisions; explain the tradeoff.")
    print("This synthetic example does not demonstrate a bug in LeRobot or FLAI.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        demo()
