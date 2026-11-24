import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIVOT_ORDER = ["First", "Last", "Middle", "Random", "Median-of-Three", "Median-of-Medians"]
PATTERN_FOR_SPEEDUP_CHARTS = "Random"   # change if you want Charts 2/3 for a different input


def read_csv(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"ERROR: '{path}' not found. Run ./main first to generate it.")
        sys.exit(1)

def chart1_worst_case_blowup(rows):
    patterns = ["Sorted", "Reverse-sorted"]
    data = {p: {} for p in patterns}
    for row in rows:
        if row["pattern"] in patterns:
            data[row["pattern"]][row["pivot"]] = float(row["time_ms"])

    pivots = [p for p in PIVOT_ORDER if all(p in data[pat] for pat in patterns)]
    x = range(len(pivots))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, pat in enumerate(patterns):
        values = [data[pat][p] for p in pivots]
        offset = (i - 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=pat)

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivots, rotation=20, ha="right")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title("Chart 1: Worst-case blowup by pivot choice\n(Sorted / Reverse-sorted input, guard OFF)")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("chart1_worst_case_blowup.png", dpi=200)
    print("Wrote chart1_worst_case_blowup.png")

def chart2_and_3_speedup_efficiency(rows):
    rows = [r for r in rows if r["pattern"] == PATTERN_FOR_SPEEDUP_CHARTS]
    if not rows:
        print(f"ERROR: no rows for pattern '{PATTERN_FOR_SPEEDUP_CHARTS}' in benchmark_results.csv")
        return

    series = defaultdict(lambda: ([], []))  # pivot -> (threads, values)
    series_eff = defaultdict(lambda: ([], []))

    for r in rows:
        t = int(r["threads"])
        series[r["pivot"]][0].append(t)
        series[r["pivot"]][1].append(float(r["speedup"]))
        series_eff[r["pivot"]][0].append(t)
        series_eff[r["pivot"]][1].append(float(r["efficiency"]))

    def plot(series_dict, ylabel, title, ideal_fn, filename):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for pivot in PIVOT_ORDER:
            if pivot not in series_dict:
                continue
            threads, values = series_dict[pivot]
            pairs = sorted(zip(threads, values))
            threads_sorted = [p[0] for p in pairs]
            values_sorted = [p[1] for p in pairs]
            ax.plot(threads_sorted, values_sorted, marker="o", label=pivot)

        all_threads = sorted({t for v in series_dict.values() for t in v[0]})
        ax.plot(all_threads, [ideal_fn(t) for t in all_threads],
                linestyle="--", color="gray", label="Ideal")

        ax.set_xlabel("Threads")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(all_threads)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=200)
        print(f"Wrote {filename}")

    plot(series, "Speedup (Tseq / Tpar)",
         f"Chart 2: Speedup vs threads ({PATTERN_FOR_SPEEDUP_CHARTS} input)",
         ideal_fn=lambda t: t,
         filename="chart2_speedup_vs_threads.png")

    plot(series_eff, "Efficiency (Speedup / Threads)",
         f"Chart 3: Parallel efficiency vs threads ({PATTERN_FOR_SPEEDUP_CHARTS} input)",
         ideal_fn=lambda t: 1.0,
         filename="chart3_efficiency_vs_threads.png")


def main():
    pivot_case_rows = read_csv("pivot_case_demo.csv")
    benchmark_rows = read_csv("benchmark_results.csv")

    chart1_worst_case_blowup(pivot_case_rows)
    chart2_and_3_speedup_efficiency(benchmark_rows)

if __name__ == "__main__":
    main()