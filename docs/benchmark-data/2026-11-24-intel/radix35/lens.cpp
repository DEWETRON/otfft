// Times createComplexFFT fwd0 for smooth lengths; prints microseconds and equivalent MFLOPS.
#include <chrono>
#include <cmath>
#include <cstdio>
#include "otfft.h"
using namespace OTFFT;
int main()
{
    const int ns[] = {3, 9, 27, 243, 2187, 19683, 59049, 5, 25, 125, 3125, 15625, 78125, 15, 45, 75, 225, 375, 1125, 3375, 480, 960, 1920, 7680, 1000, 3000, 6000, 12000, 100000, 10080, 1024, 4096, 65536};
    std::printf("N,us,mflops\n");
    for (int n : ns) {
        auto f = Factory::createComplexFFT(n);
        auto* x = (complex_t*)simd_malloc(n * sizeof(complex_t));
        for (int i = 0; i < n; i++) { x[i].Re = (i % 7) - 3; x[i].Im = (i % 5) - 2; }
        typedef std::chrono::steady_clock C;
        int loops = 1;
        for (;;) { auto t = C::now(); for (int j = 0; j < loops; j++) f->fwd0(x); double ms = std::chrono::duration<double, std::milli>(C::now() - t).count(); if (ms > 30 || loops > (1 << 20)) break; loops *= 2; }
        double best = 1e30;
        for (int r = 0; r < 7; r++) { auto t = C::now(); for (int j = 0; j < loops; j++) f->fwd0(x); double us = std::chrono::duration<double, std::micro>(C::now() - t).count() / loops; if (us < best) best = us; }
        std::printf("%d,%.2f,%.0f\n", n, best, 5.0 * n * std::log2((double)n) / best);
        simd_free(x);
    }
}
