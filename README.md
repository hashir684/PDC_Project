# PDC Semester Project #20 — Parallel Quick Sort (Pivot-Case Handling) using OpenMP

## Group Members
- Zayan Ahmed — 23K-0891
- Muhammad Hashir Khan — 23K-0684
- Muhammad Hammad Uddin — 23K-0863

## Overview
Parallel Quick Sort implemented with OpenMP tasking (`omp task` / `omp taskwait`),
covering all six pivot selection strategies — First, Last, Middle, Random,
Median-of-Three, and Median-of-Medians — and handling every pivot case
(best, average, worst, and equal-keys) via Hoare partitioning with equal-key
stopping and a Heap Sort worst-case guard.

## Build

g++ -O2 -std=c++11 -fopenmp main.cpp -o main

## Run

./main

Runs correctness tests, the pivot-case demo, and the parallel benchmark,
and writes `pivot_case_demo.csv` and `benchmark_results.csv`.

## Generate Charts

python plot_charts.py

Reads the two CSV files above and produces:
- `chart1_worst_case_blowup.png`
- `chart2_speedup_vs_threads.png`
- `chart3_efficiency_vs_threads.png`