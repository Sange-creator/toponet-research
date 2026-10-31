"""Training / sampler.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

import random


class BalancedRouteSampler:
    """Each bag uses every represented start once; per-start route bags avoid repeats.
    Balances episode starts, NOT distance, elapsed time, road coverage or env steps.
    Supply ONLY the eventual training split, not the full unsplit candidate bank.
    """

    def __init__(self, routes, seed=303):
        self.rng = random.Random(seed)
        self.by_start = {}
        ids = set()
        for r in routes:
            rid = r["route_id"]
            if rid in ids:
                raise ValueError("Duplicate route ID")
            ids.add(rid)
            self.by_start.setdefault(int(r["start_site_id"]), []).append(rid)
        if not self.by_start:
            raise ValueError("Empty training split")
        for start in self.by_start:
            self.by_start[start].sort()
        self.starts = sorted(self.by_start)
        self.start_bag = []
        self.route_bags = {s: [] for s in self.starts}

    def next_route_id(self):
        if not self.start_bag:
            self.start_bag = self.starts.copy()
            self.rng.shuffle(self.start_bag)
        start = self.start_bag.pop()
        if not self.route_bags[start]:
            self.route_bags[start] = self.by_start[start].copy()
            self.rng.shuffle(self.route_bags[start])
        return self.route_bags[start].pop()
