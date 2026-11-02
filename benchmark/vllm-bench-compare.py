#!/usr/bin/env python3
"""Compare two vllm-bench.sh runs.

Usage: python3 vllm-bench-compare.py <before-label> <after-label> [bench_root]
Default bench_root: /srv/vllm/benchmark/results
"""
import glob
import json
import os
import re
import statistics
import sys

root = sys.argv[3] if len(sys.argv) > 3 else "/srv/vllm/benchmark/results"
if len(sys.argv) < 3:
    sys.exit(__doc__)
before_label, after_label = sys.argv[1], sys.argv[2]

# metric key, display name, higher_is_better
METRICS = [
    ("output_throughput", "Total output tok/s", True),
    ("median_tpot_ms", "Per-user decode ms/token", False),
    ("median_ttft_ms", "Median time to first token ms", False),
    ("p99_ttft_ms", "p99 time to first token ms", False),
]


def load(label):
    """Return {(scenario, concurrency): {metric: [values across reps]}}."""
    data = {}
    files = glob.glob(os.path.join(root, label, "*_c*_r*.json"))
    if not files:
        sys.exit(f"No results found in {os.path.join(root, label)}")
    for path in files:
        m = re.match(r"(\w+)_c(\d+)_r(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            r = json.load(f)
        if r.get("failed"):
            print(f"WARNING: {path} has {r['failed']} failed requests")
        key = (m.group(1), int(m.group(2)))
        bucket = data.setdefault(key, {k: [] for k, _, _ in METRICS})
        for k, _, _ in METRICS:
            if k in r:
                bucket[k].append(r[k])
    return data


def spread(vals):
    """Run-to-run spread as % of the median: a rough noise floor."""
    med = statistics.median(vals)
    return 0.0 if med == 0 or len(vals) < 2 else (max(vals) - min(vals)) / med * 100


before, after = load(before_label), load(after_label)
print(f"\n{before_label}  vs  {after_label}   (medians across reps; spread = run-to-run noise)\n")
hdr = f"{'scenario':<9}{'conc':>5}  {'metric':<31}{'before':>10}{'after':>10}{'change':>9}{'noise':>8}"
print(hdr)
print("-" * len(hdr))

for key in sorted(set(before) & set(after)):
    scen, conc = key
    for k, name, higher_better in METRICS:
        b, a = before[key][k], after[key][k]
        if not b or not a:
            continue
        bm, am = statistics.median(b), statistics.median(a)
        change = (am - bm) / bm * 100 if bm else 0.0
        better = change > 0 if higher_better else change < 0
        noise = max(spread(b), spread(a))
        flag = ""
        if abs(change) > noise:
            flag = "  better" if better else "  WORSE"
        print(f"{scen:<9}{conc:>5}  {name:<31}{bm:>10.1f}{am:>10.1f}{change:>+8.1f}%{noise:>7.1f}%{flag}")
    print()

print("A change only counts if it is bigger than the noise column (shown as better/WORSE).")
