// OpenMP break-even per transform kind: time of one forward transform for several thread counts.
// usage: thr KIND T1,T2,... LOG_MIN LOG_MAX     KIND: pow2 | mixed15 | mixed3 | real | dct | bluestein
#include "otfft.h"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); }
static bool isPrime(long n) { if (n < 2) return false; for (long d = 2; d * d <= n; ++d) if (n % d == 0) return false; return true; }
static long nextPrime(long n) { while (!isPrime(n)) ++n; return n; }

template <class F> static double timeit(F f)
{
    f(); f();
    std::vector<double> v;
    for (int rep = 0; rep < 11; ++rep) {
        int iters = 1; double t;
        for (;;) { double t0 = now(); for (int i = 0; i < iters; ++i) f(); t = now() - t0; if (t > 4000) break; iters *= 2; }
        v.push_back(t / iters);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv)
{
    if (argc < 5) return 1;
    const std::string kind = argv[1];
    std::vector<int> threads;
    { std::stringstream ss(argv[2]); std::string t; while (std::getline(ss, t, ',')) threads.push_back(atoi(t.c_str())); }
    const int lmin = atoi(argv[3]), lmax = atoi(argv[4]);
    for (int l = lmin; l <= lmax; ++l) {
        long n = 0;
        if (kind == "pow2" || kind == "real" || kind == "dct") n = 1L << l;
        else if (kind == "mixed15") n = 15L << l;
        else if (kind == "mixed3") n = 3L << l;
        else if (kind == "bluestein") n = nextPrime(1L << l);
        if (n > (1L << 24)) break;
        printf("%s n=%ld (2^%d)", kind.c_str(), n, l);
        double t1 = 0;
        for (int T : threads) {
            omp_set_num_threads(T);
            double tmin = 1e300, tmax = 0;
            for (int plan = 0; plan < 4; ++plan) {
            double t = 0;
            void* junk = OTFFT::simd_malloc(64 + plan * 1088);   // shifts the addresses of the next allocations
            if (kind == "real") {
                double* x = (double*)OTFFT::simd_malloc(n * sizeof(double));
                OTFFT::complex_t* y = (OTFFT::complex_t*)OTFFT::simd_malloc((n + 2) * sizeof(OTFFT::complex_t));
                for (long i = 0; i < n; ++i) x[i] = (i * 7) % 11 - 5.0;
                auto f = OTFFT::Factory::createRealFFT((int)n);
                t = timeit([&] { f->fwd0(x, y); });
                OTFFT::simd_free(x); OTFFT::simd_free(y);
            } else if (kind == "dct") {
                double* x = (double*)OTFFT::simd_malloc(n * sizeof(double));
                for (long i = 0; i < n; ++i) x[i] = (i * 7) % 11 - 5.0;
                auto f = OTFFT::Factory::createDCT((int)n);
                t = timeit([&] { f->fwd0(x); });
                OTFFT::simd_free(x);
            } else {
                OTFFT::complex_t* x = (OTFFT::complex_t*)OTFFT::simd_malloc(n * sizeof(OTFFT::complex_t));
                for (long i = 0; i < n; ++i) x[i] = OTFFT::complex_t((i * 7) % 11 - 5.0, (i * 5) % 13 - 6.0);
                auto f = OTFFT::Factory::createComplexFFT((int)n);
                t = timeit([&] { f->fwd0(x); });
                OTFFT::simd_free(x);
            }
            OTFFT::simd_free(junk);
            tmin = std::min(tmin, t); tmax = std::max(tmax, t);
            }
            double t = tmin;
            if (T == threads[0]) t1 = t;
            printf("  T%d %9.1f us (%.2fx, spread %.2f)", T, t, t1 / t, tmax / tmin);
            fflush(stdout);
        }
        printf("\n");
    }
}
