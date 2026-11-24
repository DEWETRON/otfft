// Lengths with the prime factor 7 (roadmap R1.5, issue #18): the mixed radix code has
// its own radix-7 butterfly. Forward and inverse transforms are compared with the
// direct DFT sum in some bins, for lengths that use the butterfly in the middle and at
// the end of the pass sequence, and for lengths above the OpenMP threshold (2^15).
#include "otfft.h"

#include "otfft_test_checks.h"

#include <cmath>
#include <vector>

namespace
{
    std::vector<OTFFT::complex_t> makeSignal(int n)
    {
        std::vector<OTFFT::complex_t> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            x[static_cast<std::size_t>(i)] = OTFFT::complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        }
        return x;
    }

    // sign -1: forward sum, +1: inverse sum (both unnormalised)
    OTFFT::complex_t directBin(const std::vector<OTFFT::complex_t>& x, long long k, int sign)
    {
        const double pi = 3.14159265358979323846;
        const long long n = static_cast<long long>(x.size());
        double re = 0;
        double im = 0;
        for (long long j = 0; j < n; ++j)
        {
            const double angle = sign * 2.0 * pi * static_cast<double>((j * k) % n) / static_cast<double>(n);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            const auto& v = x[static_cast<std::size_t>(j)];
            re += v.Re * c - v.Im * s;
            im += v.Re * s + v.Im * c;
        }
        return OTFFT::complex_t(re, im);
    }

    void checkLength(int n)
    {
        CAPTURE(n);
        const auto input = makeSignal(n);
        auto fft = OTFFT::Factory::createComplexFFT(n);
        const long long bins[] = {0, 1, 2, 3, n / 7, n / 2, n - 3, n - 1};

        auto spectrum = input;
        fft->fwd0(OTFFT::complex_vector{spectrum.data()});
        for (long long k : bins)
        {
            if (k < 0 || k >= n) continue;
            const auto expected = directBin(input, k, -1);
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Re - expected.Re, 1e-9 * n);
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Im - expected.Im, 1e-9 * n);
        }

        auto inverse = input;
        fft->inv(OTFFT::complex_vector{inverse.data()});
        for (long long k : bins)
        {
            if (k < 0 || k >= n) continue;
            const auto expected = directBin(input, k, +1);
            CHECK_SMALL(inverse[static_cast<std::size_t>(k)].Re - expected.Re, 1e-9 * n);
            CHECK_SMALL(inverse[static_cast<std::size_t>(k)].Im - expected.Im, 1e-9 * n);
        }

        // fwd scales by 1/N, inv does not: together the identity
        auto roundtrip = input;
        fft->fwd(OTFFT::complex_vector{roundtrip.data()});
        fft->inv(OTFFT::complex_vector{roundtrip.data()});
        for (std::size_t i = 0; i < roundtrip.size(); i += 53)
        {
            CHECK_SMALL(roundtrip[i].Re - input[i].Re, 1e-9);
            CHECK_SMALL(roundtrip[i].Im - input[i].Im, 1e-9);
        }
    }
}

TEST_SUITE("otfft_radix7_test")
{

TEST_CASE("TestRadix7Only")
{
    // the butterfly alone, as the last and as the middle pass
    for (int n : {7, 49, 343, 2401})
    {
        checkLength(n);
    }
}

TEST_CASE("TestRadix7WithOtherFactors")
{
    // with 2, 4, 8, 3 and 5 before and after it
    for (int n : {14, 28, 56, 112, 21, 63, 35, 105, 70, 140, 280, 245, 315, 1260, 10080})
    {
        checkLength(n);
    }
}

TEST_CASE("TestRadix7AboveOpenMpThreshold")
{
    // N >= 2^15 runs the OpenMP versions of the passes
    for (int n : {7 * 8192, 7 * 7 * 7 * 7 * 7 * 7, 3 * 5 * 7 * 512})
    {
        checkLength(n);
    }
}

}
