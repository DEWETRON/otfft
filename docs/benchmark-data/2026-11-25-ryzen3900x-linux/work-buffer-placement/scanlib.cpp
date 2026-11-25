// time of fwd0 for the given lengths with the library as it is (one thread); prints "N us"
#include "otfft.h"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); }
int main(int argc, char** argv)
{
    omp_set_num_threads(1);
    const std::string kind = argv[1];
    for (int a = 2; a < argc; ++a) {
        const long n = atol(argv[a]);
        double best = 1e30;
        if (kind == "complex") {
            OTFFT::complex_t* x = (OTFFT::complex_t*)OTFFT::simd_malloc(n * sizeof(OTFFT::complex_t));
            for (long i = 0; i < n; ++i) x[i] = OTFFT::complex_t((i * 7) % 11 - 5.0, (i * 5) % 13 - 6.0);
            auto f = OTFFT::Factory::createComplexFFT((int)n);
            for (int rep = 0; rep < 40; ++rep) { const double t0 = now(); f->fwd0(x); best = std::min(best, now() - t0); }
            OTFFT::simd_free(x);
        } else if (kind == "real") {
            double* x = (double*)OTFFT::simd_malloc(n * sizeof(double));
            OTFFT::complex_t* y = (OTFFT::complex_t*)OTFFT::simd_malloc((n + 2) * sizeof(OTFFT::complex_t));
            for (long i = 0; i < n; ++i) x[i] = (i * 7) % 11 - 5.0;
            auto f = OTFFT::Factory::createRealFFT((int)n);
            for (int rep = 0; rep < 40; ++rep) { const double t0 = now(); f->fwd0(x, y); best = std::min(best, now() - t0); }
            OTFFT::simd_free(x); OTFFT::simd_free(y);
        } else if (kind == "dct") {
            double* x = (double*)OTFFT::simd_malloc(n * sizeof(double));
            for (long i = 0; i < n; ++i) x[i] = (i * 7) % 11 - 5.0;
            auto f = OTFFT::Factory::createDCT((int)n);
            for (int rep = 0; rep < 40; ++rep) { const double t0 = now(); f->fwd0(x); best = std::min(best, now() - t0); }
            OTFFT::simd_free(x);
        }
        printf("%ld %.2f\n", n, best);
    }
}
