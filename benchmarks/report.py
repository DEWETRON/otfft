#!/usr/bin/env python3
"""Turn otfft_bench CSV files into Markdown tables.

usage: report.py bench-t1.csv [bench-t12.csv ...] > tables.md

Each table shows equivalent MFLOPS (higher is better; 5 N log2 N for complex,
2.5 N log2 N for real and DCT). The best library per row is in bold.
"""
import csv
import sys
from collections import defaultdict

MAIN = ["OTFFT", "FFTW-measure", "FFTW-estimate", "PocketFFT", "KissFFT"]
BACKENDS = ["OTFFT-sse2", "OTFFT-avx", "OTFFT-avx2", "OTFFT"]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def cell(r):
    if r is None:
        return "–"
    if r["status"] == "ok":
        return float(r["mflops"])
    if r["status"] == "invalid":
        return "wrong"
    if r["status"] == "too_slow":
        return "slow"
    return r["status"]


def fmt(v, best):
    if isinstance(v, float):
        s = f"{v:,.0f}"
        return f"**{s}**" if best is not None and v == best else s
    return v


def table(rows, kind, category, libs, caption):
    sel = [r for r in rows if r["kind"] == kind and r["category"] in category]
    if not sel:
        return ""
    by = defaultdict(dict)
    for r in sel:
        by[int(r["n"])][r["library"]] = r
    present = [l for l in libs if any(l in d for d in by.values())]
    if not present:
        return ""
    out = [f"**{caption}**", "", "| N | " + " | ".join(present) + " |",
           "| ---: | " + " | ".join("---:" for _ in present) + " |"]
    for n in sorted(by):
        vals = [cell(by[n].get(l)) for l in present]
        nums = [v for v in vals if isinstance(v, float)]
        best = max(nums) if nums else None
        out.append(f"| {n:,} | " + " | ".join(fmt(v, best) for v in vals) + " |")
    out.append("")
    return "\n".join(out)


SINGLE = ("PocketFFT", "KissFFT")


def main(paths):
    first = None
    for path in paths:
        rows = load(path)
        if not rows:
            continue
        threads = rows[0]["threads"]
        if first is None:
            first = rows
        else:
            # PocketFFT and KissFFT are single-threaded: reuse the first file's numbers
            have = {(r["library"], r["kind"], r["n"]) for r in rows}
            rows += [r for r in first if r["library"] in SINGLE and (r["library"], r["kind"], r["n"]) not in have]
        print(f"### {threads} thread{'s' if threads != '1' else ''}\n")
        # OTFFT/FFTW use the requested thread count; PocketFFT/KissFFT are single-threaded
        for kind, title in (("complex", "Complex FFT"), ("real", "Real FFT (real to complex)"),
                            ("dct", "DCT-II")):
            for cats, label in ((["pow2"], "powers of two"), (["smooth"], "smooth lengths"),
                                (["prime"], "prime lengths")):
                libs = MAIN + (["OTFFT-bluestein"] if kind == "complex" and cats != ["pow2"] else [])
                t = table(rows, kind, cats, libs, f"{title}, {label}, MFLOPS")
                if t:
                    print(t)
        t = table(rows, "complex", ["pow2"], BACKENDS, "OTFFT backends, complex FFT, powers of two, MFLOPS")
        if t:
            print(t)
        # plan time
        sel = [r for r in rows if r["kind"] == "complex" and r["category"] == "pow2"
               and r["library"] in ("OTFFT", "FFTW-measure", "FFTW-estimate", "PocketFFT", "KissFFT")]
        by = defaultdict(dict)
        for r in sel:
            by[int(r["n"])][r["library"]] = float(r["plan_us"])
        libs = [l for l in MAIN if any(l in d for d in by.values())]
        if by:
            print("**Plan creation, complex FFT, powers of two, microseconds**\n")
            print("| N | " + " | ".join(libs) + " |")
            print("| ---: | " + " | ".join("---:" for _ in libs) + " |")
            for n in sorted(by):
                print(f"| {n:,} | " + " | ".join(f"{by[n][l]:,.0f}" if l in by[n] else "–" for l in libs) + " |")
            print()
        bad = [r for r in rows if r["status"] == "invalid"]
        if bad:
            print("**Wrong results (not timed)**\n")
            print("| Library | Kind | N | Relative error |")
            print("| --- | --- | ---: | ---: |")
            for r in bad:
                print(f"| {r['library']} | {r['kind']} | {int(r['n']):,} | {float(r['rel_error']):.3g} |")
            print()
        scaled = {(r["library"], r["kind"]): r["scale"] for r in rows
                  if r["status"] == "ok" and float(r["scale"]) != 1.0}
        for (lib, kind), sc in sorted(scaled.items()):
            print(f"Note: {lib} {kind} output is {sc} times PocketFFT's (convention only).\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
