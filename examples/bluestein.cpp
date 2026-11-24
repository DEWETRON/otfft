// Bluestein FFT: any length, including primes, in O(N log N).
//
// createComplexFFT also switches to Bluestein for lengths with a prime factor
// above 100, so this example shows both entry points.

#include "otfft.h"

#include <cmath>
#include <cstdio>

using OTFFT::complex_t;
using OTFFT::simd_free;
using OTFFT::simd_malloc;

// One bin of the direct DFT sum to check against (O(N), forward, not scaled).
static complex_t directBin(const complex_t* x, int N, int k)
{
    const double pi = 3.14159265358979323846;
    double re = 0;
    double im = 0;
    for (int n = 0; n < N; ++n)
    {
        const double angle = -2.0 * pi * static_cast<double>((static_cast<long long>(n) * k) % N) / N;
        re += x[n].Re * std::cos(angle) - x[n].Im * std::sin(angle);
        im += x[n].Re * std::sin(angle) + x[n].Im * std::cos(angle);
    }
    return complex_t(re, im);
}

static int run(const char* name, OTFFT::ComplexFFTPtr& fft, int N)
{
    complex_t* x = static_cast<complex_t*>(simd_malloc(N * sizeof(complex_t)));
    complex_t* original = static_cast<complex_t*>(simd_malloc(N * sizeof(complex_t)));
    for (int n = 0; n < N; ++n)
    {
        x[n] = complex_t(((n * 7) % 11) - 5.0, ((n * 5) % 13) - 6.0);
        original[n] = x[n];
    }

    fft->fwd0(x); // no scaling, to compare with the direct sum

    int failures = 0;
    double maxDifference = 0;
    const int bins[] = {0, 1, 2, N / 3, N - 1};
    for (int k : bins)
    {
        const complex_t d = directBin(original, N, k);
        maxDifference = std::fmax(maxDifference, std::fabs(x[k].Re - d.Re) + std::fabs(x[k].Im - d.Im));
    }
    std::printf("%s, N = %d: largest difference to the direct sum (5 bins): %.3g\n", name, N, maxDifference);
    if (maxDifference > 1e-8)
    {
        ++failures;
    }

    fft->invn(x); // divides by N
    double maxError = 0;
    for (int n = 0; n < N; ++n)
    {
        maxError = std::fmax(maxError, std::fabs(x[n].Re - original[n].Re) + std::fabs(x[n].Im - original[n].Im));
    }
    std::printf("%s, N = %d: largest error after fwd0 and invn: %.3g\n", name, N, maxError);
    if (maxError > 1e-9)
    {
        ++failures;
    }

    simd_free(x);
    simd_free(original);
    return failures;
}

int main()
{
    int failures = 0;

    // 10007 is prime.
    auto bluestein = OTFFT::Factory::createBluesteinFFT(10007);
    failures += run("createBluesteinFFT", bluestein, 10007);

    // createComplexFFT chooses Bluestein by itself here: 2018 = 2 * 1009 and 1009 is above 100.
    auto automatic = OTFFT::Factory::createComplexFFT(2018);
    failures += run("createComplexFFT ", automatic, 2018);

    std::printf(failures == 0 ? "bluestein: ok\n" : "bluestein: FAILED (%d checks)\n", failures);
    return failures == 0 ? 0 : 1;
}
