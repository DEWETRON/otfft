#!/usr/bin/env python3
"""Compare this fork's OTFFT with upstream OTFFT in the same session.

usage: compare_upstream.py fork.csv upstream.csv [fork2.csv upstream2.csv ...]

Each pair is one run of otfft_bench and one run of the upstream build of the
same harness (OTFFT_UPSTREAM_DIR). With several pairs the median over the pairs
is used and the spread (min to max) is printed. Output is Markdown.

Ratio = fork MFLOPS / upstream MFLOPS (above 1 means the fork is faster).
"""
import csv
import math
import statistics
import sys
from collections import defaultdict

FORK, UP = "OTFFT", "OTFFT-upstream"


def load(path):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok":
                d[(r["library"], r["kind"], int(r["n"]))] = float(r["mflops"])
    return d


def geomean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def main(paths):
    if len(paths) < 2 or len(paths) % 2:
        sys.exit(__doc__)
    pairs = [(load(paths[i]), load(paths[i + 1])) for i in range(0, len(paths), 2)]
    ratios = defaultdict(list)
    for fk, up in pairs:
        for (lib, kind, n), v in fk.items():
            if lib == FORK and (UP, kind, n) in up:
                ratios[(kind, n)].append(v / up[(UP, kind, n)])
    kinds = sorted({k for k, _ in ratios})
    for kind in kinds:
        rows = sorted((n, r) for (k, n), r in ratios.items() if k == kind)
        meds = [statistics.median(r) for _, r in rows]
        print(f"**{kind}** (geometric mean of medians {geomean(meds):.2f}, "
              f"{len(pairs)} run{'s' if len(pairs) > 1 else ''})\n")
        print("| N | fork / upstream | min | max |")
        print("| ---: | ---: | ---: | ---: |")
        for (n, r), m in zip(rows, meds):
            print(f"| {n:,} | {m:.2f} | {min(r):.2f} | {max(r):.2f} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
