// Checks of createComplexFFT(n) against the direct DFT sum, for the mixed radix tests.
#pragma once

#include "otfft.h"

#include "otfft_test_checks.h"

#include <cmath>
#include <vector>

namespace OTFFT
{
namespace test
{
    // The same signal for every call with the same n.
    inline std::vector<complex_t> makeSignal(int n)
    {
        std::vector<complex_t> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            x[static_cast<std::size_t>(i)] = complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        }
        return x;
    }

    // One bin k of the direct DFT sum (O(n)); sign -1: forward, +1: inverse, both unnormalised.
    inline complex_t directBin(const std::vector<complex_t>& x, long long k, int sign)
    {
        const double pi = 3.14159265358979323846;
        const long long n = static_cast<long long>(x.size());
        double re = 0;
        double im = 0;
        for (long long j = 0; j < n; ++j)
        {
            // (j*k) % n in 64 bit: j*k does not fit in 32 bit for large n
            const double angle = sign * 2.0 * pi * static_cast<double>((j * k) % n) / static_cast<double>(n);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            const auto& v = x[static_cast<std::size_t>(j)];
            re += v.Re * c - v.Im * s;
            im += v.Re * s + v.Im * c;
        }
        return complex_t(re, im);
    }

    // fwd0 and inv of createComplexFFT(n) agree with the direct sum in some bins, and
    // fwd followed by inv gives back the input.
    inline void checkComplexLength(int n)
    {
        CAPTURE(n);
        const auto input = makeSignal(n);
        auto fft = Factory::createComplexFFT(n);
        const long long bins[] = {0, 1, 2, 3, n / 7, n / 2, n - 3, n - 1};

        auto spectrum = input;
        fft->fwd0(complex_vector{spectrum.data()});
        for (long long k : bins)
        {
            if (k < 0 || k >= n) continue;
            const auto expected = directBin(input, k, -1);
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Re - expected.Re, 1e-9 * n);
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Im - expected.Im, 1e-9 * n);
        }

        auto inverse = input;
        fft->inv(complex_vector{inverse.data()});
        for (long long k : bins)
        {
            if (k < 0 || k >= n) continue;
            const auto expected = directBin(input, k, +1);
            CHECK_SMALL(inverse[static_cast<std::size_t>(k)].Re - expected.Re, 1e-9 * n);
            CHECK_SMALL(inverse[static_cast<std::size_t>(k)].Im - expected.Im, 1e-9 * n);
        }

        // fwd scales by 1/N and inv does not: together the identity
        auto roundtrip = input;
        fft->fwd(complex_vector{roundtrip.data()});
        fft->inv(complex_vector{roundtrip.data()});
        for (std::size_t i = 0; i < roundtrip.size(); i += 53)
        {
            CHECK_SMALL(roundtrip[i].Re - input[i].Re, 1e-9);
            CHECK_SMALL(roundtrip[i].Im - input[i].Im, 1e-9);
        }
    }
}
}
