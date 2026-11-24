// Complex FFT: find the two tones in a signal, then go back to the signal.
//
// The program returns 0 if every check passes, so ctest can run it.

#include "otfft.h"

#include <cmath>
#include <cstdio>

using OTFFT::complex_t;
using OTFFT::simd_free;
using OTFFT::simd_malloc;

int main()
{
    const int N = 1024;
    const double pi = 3.14159265358979323846;

    // Buffers must be SIMD aligned: use simd_malloc and simd_free.
    complex_t* x = static_cast<complex_t*>(simd_malloc(N * sizeof(complex_t)));
    complex_t* original = static_cast<complex_t*>(simd_malloc(N * sizeof(complex_t)));

    // Two complex tones: 50 cycles with amplitude 1.0 and 200 cycles with amplitude 0.5.
    for (int n = 0; n < N; ++n)
    {
        const double a = 2.0 * pi * 50.0 * n / N;
        const double b = 2.0 * pi * 200.0 * n / N;
        x[n] = complex_t(std::cos(a) + 0.5 * std::cos(b), std::sin(a) + 0.5 * std::sin(b));
        original[n] = x[n];
    }

    // The length is given on creation. Use one object per thread: it owns a work buffer.
    auto fft = OTFFT::Factory::createComplexFFT(N);

    // fwd divides by N, so a tone of amplitude A gives the value A in its bin.
    fft->fwd(x);

    int failures = 0;
    std::printf("bin   amplitude\n");
    for (int k = 0; k < N; ++k)
    {
        const double amplitude = std::sqrt(x[k].Re * x[k].Re + x[k].Im * x[k].Im);
        if (amplitude > 0.01)
        {
            std::printf("%4d  %.6f\n", k, amplitude);
        }
        const double expected = (k == 50) ? 1.0 : (k == 200) ? 0.5 : 0.0;
        if (std::fabs(amplitude - expected) > 1e-9)
        {
            ++failures;
        }
    }

    // inv does not scale, so inv(fwd(x)) returns x.
    fft->inv(x);
    double maxError = 0;
    for (int n = 0; n < N; ++n)
    {
        maxError = std::fmax(maxError, std::fabs(x[n].Re - original[n].Re));
        maxError = std::fmax(maxError, std::fabs(x[n].Im - original[n].Im));
    }
    std::printf("largest error after fwd and inv: %.3g\n", maxError);
    if (maxError > 1e-12)
    {
        ++failures;
    }

    // The same object takes another length with setup(). 1000 = 2^3 * 5^3 uses the mixed radix code.
    fft->setup(1000);
    for (int n = 0; n < 1000; ++n)
    {
        x[n] = complex_t(n % 2 == 0 ? 1.0 : -1.0, 0.0); // a tone at the highest frequency
    }
    fft->fwd(x);
    std::printf("N = 1000, alternating signal: bin 500 = %.6f (expected 1)\n", x[500].Re);
    if (std::fabs(x[500].Re - 1.0) > 1e-12)
    {
        ++failures;
    }

    simd_free(x);
    simd_free(original);

    std::printf(failures == 0 ? "complex_fft: ok\n" : "complex_fft: FAILED (%d checks)\n", failures);
    return failures == 0 ? 0 : 1;
}
