// DCT-II: in place, not orthogonalised, N must be even.
//
// fwd0 computes X[k] = sum over n of x[n] * cos(pi * (n + 1/2) * k / N), which is half
// of what FFTW's REDFT10 computes. fwd divides that by N.

#include "otfft.h"

#include <cmath>
#include <cstdio>

using OTFFT::simd_free;
using OTFFT::simd_malloc;

int main()
{
    const int N = 64;
    const double pi = 3.14159265358979323846;

    double* x = static_cast<double*>(simd_malloc(N * sizeof(double)));
    double* original = static_cast<double*>(simd_malloc(N * sizeof(double)));

    // A smooth ramp: most of its energy ends up in the first coefficients.
    for (int n = 0; n < N; ++n)
    {
        x[n] = static_cast<double>(n) / N;
        original[n] = x[n];
    }

    auto dct = OTFFT::Factory::createDCT(N);
    dct->fwd0(x);

    int failures = 0;

    // Compare with the sum above.
    double maxDifference = 0;
    for (int k = 0; k < N; ++k)
    {
        double sum = 0;
        for (int n = 0; n < N; ++n)
        {
            sum += original[n] * std::cos(pi * (n + 0.5) * k / N);
        }
        maxDifference = std::fmax(maxDifference, std::fabs(sum - x[k]));
    }
    std::printf("largest difference to the direct sum: %.3g\n", maxDifference);
    if (maxDifference > 1e-10)
    {
        ++failures;
    }

    // Energy compaction: the first 4 of 64 coefficients carry nearly all of the energy.
    // (The weights make the coefficients orthogonal: 1 for k = 0, 2 for the others.)
    double total = 0;
    double head = 0;
    for (int k = 0; k < N; ++k)
    {
        const double e = x[k] * x[k] * (k == 0 ? 1.0 : 2.0);
        total += e;
        if (k < 4)
        {
            head += e;
        }
    }
    std::printf("share of the energy in coefficients 0 to 3: %.4f\n", head / total);
    if (head / total < 0.99)
    {
        ++failures;
    }

    // fwd divides by N and inv does not scale, so inv(fwd(x)) returns x.
    dct->fwd(original);
    dct->inv(original);
    double maxError = 0;
    for (int n = 0; n < N; ++n)
    {
        maxError = std::fmax(maxError, std::fabs(original[n] - static_cast<double>(n) / N));
    }
    std::printf("largest error after fwd and inv: %.3g\n", maxError);
    if (maxError > 1e-12)
    {
        ++failures;
    }

    simd_free(x);
    simd_free(original);

    std::printf(failures == 0 ? "dct: ok\n" : "dct: FAILED (%d checks)\n", failures);
    return failures == 0 ? 0 : 1;
}
