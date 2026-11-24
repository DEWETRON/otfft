// Real FFT: the spectrum of a real signal, and back.
//
// The output of fwd holds all N values of the conjugate-symmetric spectrum (not N/2+1).
// The input of inv is destroyed. N must be even.

#include "otfft.h"

#include <cmath>
#include <cstdio>

using OTFFT::complex_t;
using OTFFT::simd_free;
using OTFFT::simd_malloc;

int main()
{
    const int N = 2000; // even, not a power of two: the mixed radix code is used
    const double pi = 3.14159265358979323846;

    double* x = static_cast<double*>(simd_malloc(N * sizeof(double)));
    double* original = static_cast<double*>(simd_malloc(N * sizeof(double)));
    complex_t* y = static_cast<complex_t*>(simd_malloc(N * sizeof(complex_t)));

    // A cosine of amplitude 3 with 40 cycles, plus an offset of 2.
    for (int n = 0; n < N; ++n)
    {
        x[n] = 2.0 + 3.0 * std::cos(2.0 * pi * 40.0 * n / N);
        original[n] = x[n];
    }

    auto rfft = OTFFT::Factory::createRealFFT(N);

    // x is only read. fwd divides by N: bin 0 is the mean, and a cosine of
    // amplitude A gives A/2 in bin k and in the mirror bin N-k.
    rfft->fwd(x, y);

    int failures = 0;
    std::printf("mean (bin 0)      : %.6f (expected 2)\n", y[0].Re);
    std::printf("bin 40            : %.6f (expected 1.5)\n", y[40].Re);
    std::printf("bin N-40 (mirror) : %.6f (expected 1.5)\n", y[N - 40].Re);
    if (std::fabs(y[0].Re - 2.0) > 1e-12 || std::fabs(y[40].Re - 1.5) > 1e-12 ||
        std::fabs(y[N - 40].Re - 1.5) > 1e-12)
    {
        ++failures;
    }

    // y is destroyed by inv.
    rfft->inv(y, x);
    double maxError = 0;
    for (int n = 0; n < N; ++n)
    {
        maxError = std::fmax(maxError, std::fabs(x[n] - original[n]));
    }
    std::printf("largest error after fwd and inv: %.3g\n", maxError);
    if (maxError > 1e-12)
    {
        ++failures;
    }

    simd_free(x);
    simd_free(original);
    simd_free(y);

    std::printf(failures == 0 ? "real_fft: ok\n" : "real_fft: FAILED (%d checks)\n", failures);
    return failures == 0 ? 0 : 1;
}
