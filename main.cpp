#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <omp.h>
using namespace std;

double wallTime() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

enum class PivotStrategy {
    FIRST, LAST, MIDDLE, RANDOM, MEDIAN_OF_THREE, MEDIAN_OF_MEDIANS
};

const vector<PivotStrategy> ALL_PIVOTS = {
    PivotStrategy::FIRST, PivotStrategy::LAST, PivotStrategy::MIDDLE,
    PivotStrategy::RANDOM, PivotStrategy::MEDIAN_OF_THREE,
    PivotStrategy::MEDIAN_OF_MEDIANS
};

string pivotName(PivotStrategy p) {
    switch (p) {
        case PivotStrategy::FIRST:             return "First";
        case PivotStrategy::LAST:              return "Last";
        case PivotStrategy::MIDDLE:            return "Middle";
        case PivotStrategy::RANDOM:            return "Random";
        case PivotStrategy::MEDIAN_OF_THREE:   return "Median-of-Three";
        case PivotStrategy::MEDIAN_OF_MEDIANS: return "Median-of-Medians";
    }
    return "Unknown";
}

struct SortOptions {
    PivotStrategy pivot;
    bool guard;
};

static int CUTOFF = 10000;

int floorLog2(int n) {
    int k = 0;
    while (n > 1) { n >>= 1; ++k; }
    return k;
}

int depthLimit(int n) {
    return 4 * floorLog2(n);
}

void insertionSort(int* a, int lo, int hi) {
    for (int i = lo + 1; i <= hi; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= lo && a[j] > key) { a[j + 1] = a[j]; --j; }
        a[j + 1] = key;
    }
}

static void siftDown(int* b, int n, int i) {
    int v = b[i];
    while (true) {
        int c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && b[c + 1] > b[c]) ++c;
        if (b[c] <= v) break;
        b[i] = b[c];
        i = c;
    }
    b[i] = v;
}

void heapSortRange(int* a, int lo, int hi) {
    int* b = a + lo;
    int n = hi - lo + 1;
    for (int i = n / 2 - 1; i >= 0; --i) siftDown(b, n, i);
    for (int end = n - 1; end > 0; --end) {
        swap(b[0], b[end]);
        siftDown(b, end, 0);
    }
}

int partitionHoare(int* a, int lo, int hi) {
    int v = a[lo];
    int i = lo, j = hi + 1;
    while (true) {
        while (a[++i] < v) if (i == hi) break;
        while (v < a[--j]) {}
        if (i >= j) break;
        swap(a[i], a[j]);
    }
    swap(a[lo], a[j]);
    return j;
}

thread_local mt19937 rng(random_device{}() + omp_get_thread_num());

int randomIndex(int lo, int hi) {
    uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}

int medianOfThreeIndex(int* a, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    if (a[mid] < a[lo])  swap(a[mid], a[lo]);
    if (a[hi]  < a[lo])  swap(a[hi],  a[lo]);
    if (a[hi]  < a[mid]) swap(a[hi],  a[mid]);
    return mid;
}

int selectKth(int* a, int lo, int hi, int k);

int medianOfMediansIndex(int* a, int lo, int hi) {
    int n = hi - lo + 1;
    if (n <= 5) {
        insertionSort(a, lo, hi);
        return lo + (n - 1) / 2;
    }
    int m = lo;
    for (int i = lo; i <= hi; i += 5) {
        int e = min(i + 4, hi);
        insertionSort(a, i, e);
        swap(a[m++], a[i + (e - i) / 2]);
    }
    return selectKth(a, lo, m - 1, lo + (m - 1 - lo) / 2);
}

int selectKth(int* a, int lo, int hi, int k) {
    while (lo < hi) {
        int p = medianOfMediansIndex(a, lo, hi);
        swap(a[lo], a[p]);
        int j = partitionHoare(a, lo, hi);
        if (j == k) return k;
        if (k < j) hi = j - 1;
        else       lo = j + 1;
    }
    return k;
}

int choosePivotIndex(int* a, int lo, int hi, PivotStrategy s) {
    switch (s) {
        case PivotStrategy::FIRST:             return lo;
        case PivotStrategy::LAST:              return hi;
        case PivotStrategy::MIDDLE:            return lo + (hi - lo) / 2;
        case PivotStrategy::RANDOM:            return randomIndex(lo, hi);
        case PivotStrategy::MEDIAN_OF_THREE:   return medianOfThreeIndex(a, lo, hi);
        case PivotStrategy::MEDIAN_OF_MEDIANS: return medianOfMediansIndex(a, lo, hi);
    }
    return lo;
}

int partitionStep(int* a, int lo, int hi, PivotStrategy s) {
    int p = choosePivotIndex(a, lo, hi, s);
    swap(a[lo], a[p]);
    return partitionHoare(a, lo, hi);
}

void quickSortSeq(int* a, int lo, int hi, SortOptions opt,
                  int depthLeft, int level, int& maxDepth) {
    while (lo < hi) {
        if (opt.guard && depthLeft == 0) {
            heapSortRange(a, lo, hi);
            return;
        }
        if (level > maxDepth) maxDepth = level;

        int p = partitionStep(a, lo, hi, opt.pivot);
        --depthLeft;
        ++level;

        if (p - lo < hi - p) {
            quickSortSeq(a, lo, p - 1, opt, depthLeft, level, maxDepth);
            lo = p + 1;
        } else {
            quickSortSeq(a, p + 1, hi, opt, depthLeft, level, maxDepth);
            hi = p - 1;
        }
    }
}

void sequentialQuickSort(vector<int>& v, SortOptions opt, int& maxDepth) {
    maxDepth = 0;
    int n = (int)v.size();
    if (n < 2) return;
    quickSortSeq(v.data(), 0, n - 1, opt, depthLimit(n), 1, maxDepth);
}

void quickSortPar(int* a, int lo, int hi, SortOptions opt, int depthLeft) {
    int ignored = 0;
    while (hi - lo + 1 > CUTOFF) {
        if (opt.guard && depthLeft == 0) {
            heapSortRange(a, lo, hi);
            lo = hi;
            break;
        }

        int p = partitionStep(a, lo, hi, opt.pivot);
        --depthLeft;

        int sLo, sHi;
        if (p - lo < hi - p) { sLo = lo;    sHi = p - 1; lo = p + 1; }
        else                 { sLo = p + 1; sHi = hi;    hi = p - 1; }

        if (sHi - sLo + 1 > CUTOFF) {
            #pragma omp task firstprivate(a, sLo, sHi, opt, depthLeft)
            quickSortPar(a, sLo, sHi, opt, depthLeft);
        } else {
            quickSortSeq(a, sLo, sHi, opt, depthLeft, 0, ignored);
        }
    }
    quickSortSeq(a, lo, hi, opt, depthLeft, 0, ignored);
    #pragma omp taskwait
}

void parallelQuickSort(vector<int>& v, SortOptions opt) {
    int n = (int)v.size();
    if (n < 2) return;
    int* a = v.data();
    int limit = depthLimit(n);
    #pragma omp parallel
    {
        #pragma omp single nowait
        quickSortPar(a, 0, n - 1, opt, limit);
    }
}

enum class InputPattern { RANDOM, SORTED, REVERSE_SORTED, MANY_DUPLICATES, ALL_EQUAL };

const vector<InputPattern> ALL_PATTERNS = {
    InputPattern::RANDOM, InputPattern::SORTED, InputPattern::REVERSE_SORTED,
    InputPattern::MANY_DUPLICATES, InputPattern::ALL_EQUAL
};

string patternName(InputPattern p) {
    switch (p) {
        case InputPattern::RANDOM:          return "Random";
        case InputPattern::SORTED:          return "Sorted";
        case InputPattern::REVERSE_SORTED:  return "Reverse-sorted";
        case InputPattern::MANY_DUPLICATES: return "Many-duplicates";
        case InputPattern::ALL_EQUAL:       return "All-equal";
    }
    return "Unknown";
}

vector<int> generateData(int n, InputPattern pattern, unsigned seed = 42) {
    vector<int> data(n);
    mt19937 gen(seed);

    switch (pattern) {
        case InputPattern::RANDOM: {
            uniform_int_distribution<int> dist(0, 1000000000);
            for (int i = 0; i < n; ++i) data[i] = dist(gen);
            break;
        }
        case InputPattern::SORTED:
            for (int i = 0; i < n; ++i) data[i] = i;
            break;
        case InputPattern::REVERSE_SORTED:
            for (int i = 0; i < n; ++i) data[i] = n - i;
            break;
        case InputPattern::MANY_DUPLICATES: {
            uniform_int_distribution<int> dist(0, 50);
            for (int i = 0; i < n; ++i) data[i] = dist(gen);
            break;
        }
        case InputPattern::ALL_EQUAL:
            fill(data.begin(), data.end(), 7);
            break;
    }
    return data;
}

struct RunResult {
    double seconds;
    int depth;
};

RunResult runSequential(vector<int>& v, SortOptions opt) {
    int depth = 0;
    double t0 = wallTime();
    sequentialQuickSort(v, opt, depth);
    double t1 = wallTime();
    return { t1 - t0, depth };
}

RunResult runParallel(vector<int>& v, SortOptions opt, int threads) {
    omp_set_num_threads(threads);
    double t0 = wallTime();
    parallelQuickSort(v, opt);
    double t1 = wallTime();
    return { t1 - t0, -1 };
}

void verifyOrDie(const vector<int>& got, const vector<int>& expected, const string& what) {
    if (got != expected) {
        cerr << "\nERROR: wrong result for " << what << "\n";
        exit(1);
    }
}

void runCorrectnessTests(int maxThreads) {
    cout << "=== 1. Correctness tests ===\n";
    int savedCutoff = CUTOFF;
    CUTOFF = 64;

    vector<int> sizes = {0, 1, 2, 3, 5, 10, 100, 1000, 20000};
    long long total = 0;

    for (int n : sizes) {
        for (auto pat : ALL_PATTERNS) {
            vector<int> base = generateData(n, pat);
            vector<int> ref = base;
            sort(ref.begin(), ref.end());

            for (auto piv : ALL_PIVOTS) {
                for (int par = 0; par <= 1; ++par) {
                    vector<int> v = base;
                    SortOptions opt = { piv, true };
                    if (par) runParallel(v, opt, maxThreads);
                    else     runSequential(v, opt);
                    string what = string(par ? "parallel" : "sequential") + ", n=" + to_string(n)
                                  + ", " + patternName(pat) + ", " + pivotName(piv);
                    verifyOrDie(v, ref, what);
                    ++total;
                }
            }
        }
    }
    CUTOFF = savedCutoff;

    cout << "All " << total << " tests passed (6 pivots x 5 inputs, sizes 0 to 20000, "
         << "sequential + parallel).\n\n";
}

string classifyCase(int depth, int n) {
    double lg = log2((double)max(n, 2));
    if (depth <= lg + 2) return "Best (even splits)";
    if (depth <= 5 * lg) return "Average O(n log n)";
    if (depth >= n / 10) return "WORST O(n^2)";
    return "Degraded";
}

void runPivotCaseDemo(int n) {
    cout << "=== 2. Pivot-case demo (sequential, guard OFF, n = " << n
         << ", ideal depth ~ " << (int)ceil(log2((double)max(n, 2))) << ") ===\n";
    cout << left << setw(18) << "Pattern" << setw(20) << "Pivot"
         << setw(12) << "Time(ms)" << setw(10) << "Depth" << "Case\n";
    cout << string(80, '-') << "\n";

    ofstream csv("pivot_case_demo.csv");
    csv << "pattern,pivot,time_ms,depth,case\n";

    for (auto pat : ALL_PATTERNS) {
        vector<int> base = generateData(n, pat);
        vector<int> ref = base;
        sort(ref.begin(), ref.end());

        for (auto piv : ALL_PIVOTS) {
            vector<int> v = base;
            SortOptions opt = { piv, false };
            RunResult r = runSequential(v, opt);
            verifyOrDie(v, ref, "pivot-case demo, " + patternName(pat) + ", " + pivotName(piv));

            cout << left << setw(18) << patternName(pat) << setw(20) << pivotName(piv)
                 << setw(12) << fixed << setprecision(2) << r.seconds * 1000.0
                 << setw(10) << r.depth << classifyCase(r.depth, n) << "\n";

            csv << patternName(pat) << "," << pivotName(piv) << ","
                << fixed << setprecision(2) << r.seconds * 1000.0 << ","
                << r.depth << "," << classifyCase(r.depth, n) << "\n";
        }
        cout << "\n";
    }
    csv.close();
    cout << "(Saved pivot_case_demo.csv for charting.)\n\n";
}

void runParallelBenchmark(int n, int runs, const vector<int>& threadCounts) {
    cout << "=== 3. Parallel benchmark (guard ON, cutoff = " << CUTOFF
         << ", n = " << n << ", avg of " << runs << " runs) ===\n";

    ofstream csv("benchmark_results.csv");
    csv << "pattern,pivot,threads,seq_time_s,par_time_s,speedup,efficiency\n";

    for (auto pat : ALL_PATTERNS) {
        vector<int> base = generateData(n, pat);
        vector<int> ref = base;
        sort(ref.begin(), ref.end());

        cout << "\nInput = " << patternName(pat) << "\n";
        cout << left << setw(20) << "Pivot" << setw(9) << "Threads"
             << setw(10) << "Seq(s)" << setw(10) << "Par(s)"
             << setw(10) << "Speedup" << "Efficiency\n";
        cout << string(72, '-') << "\n";

        for (auto piv : ALL_PIVOTS) {
            SortOptions opt = { piv, true };

            double seqSum = 0;
            for (int r = 0; r < runs; ++r) {
                vector<int> v = base;
                seqSum += runSequential(v, opt).seconds;
                verifyOrDie(v, ref, "benchmark sequential " + pivotName(piv));
            }
            double seqTime = seqSum / runs;

            for (int t : threadCounts) {
                double parSum = 0;
                for (int r = 0; r < runs; ++r) {
                    vector<int> v = base;
                    parSum += runParallel(v, opt, t).seconds;
                    verifyOrDie(v, ref, "benchmark parallel " + pivotName(piv));
                }
                double parTime = parSum / runs;
                double speedup = seqTime / parTime;
                double efficiency = speedup / t;

                cout << left << setw(20) << pivotName(piv) << setw(9) << t
                     << setw(10) << fixed << setprecision(4) << seqTime
                     << setw(10) << parTime
                     << setw(10) << setprecision(2) << speedup
                     << efficiency << "\n";

                csv << patternName(pat) << "," << pivotName(piv) << "," << t << ","
                    << fixed << setprecision(6) << seqTime << ","
                    << parTime << "," << setprecision(4) << speedup << ","
                    << efficiency << "\n";
            }
        }
    }
    csv.close();
    cout << "\n(Saved benchmark_results.csv for charting.)\n\n";
}

int main() {
    const int maxThreads = omp_get_max_threads();
    cout << "Parallel Quick Sort - all pivot cases (OpenMP tasks)\n"
         << "Max threads available: " << maxThreads << "\n\n";

    vector<int> threadCounts;
    for (int t : {1, 2, 4, 8}) if (t <= maxThreads) threadCounts.push_back(t);
    if (threadCounts.empty() || threadCounts.back() != maxThreads) threadCounts.push_back(maxThreads);

    int demoN         = 50000;
    int benchmarkSize = 10000000;
    int runs          = 3;

    runCorrectnessTests(maxThreads);
    runPivotCaseDemo(demoN);
    runParallelBenchmark(benchmarkSize, runs, threadCounts);

    return 0;
}