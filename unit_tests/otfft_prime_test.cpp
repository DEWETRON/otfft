// Prime and large-prime-factor lengths with createComplexFFT (roadmap R1.4,
// issues #16 and #17).
#include "otfft.h"

#include "otfft_test_checks.h"

#include <chrono>
#include <cmath>
#include <vector>

namespace
{
    // A test signal that is the same for every call with the same n.
    std::vector<OTFFT::complex_t> makeSignal(int n)
    {
        std::vector<OTFFT::complex_t> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            x[static_cast<std::size_t>(i)] = OTFFT::complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        }
        return x;
    }

    // Direct DFT sum for one bin k (O(n)), unnormalised forward transform.
    OTFFT::complex_t directBin(const std::vector<OTFFT::complex_t>& x, int k)
    {
        const double pi = 3.14159265358979323846;
        const long long n = static_cast<long long>(x.size());
        double re = 0;
        double im = 0;
        for (long long j = 0; j < n; ++j)
        {
            // (j*k) % n in 64 bit: j*k does not fit in 32 bit for n above 46,340
            const double angle = -2.0 * pi * static_cast<double>((j * k) % n) / static_cast<double>(n);
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            const auto& v = x[static_cast<std::size_t>(j)];
            re += v.Re * c - v.Im * s;
            im += v.Re * s + v.Im * c;
        }
        return OTFFT::complex_t(re, im);
    }

    // fwd0 of createComplexFFT(n) agrees with the direct sum in some bins, and fwd
    // followed by inv gives back the input.
    void checkComplex(int n)
    {
        const auto input = makeSignal(n);
        auto spectrum = input;
        auto fft = OTFFT::Factory::createComplexFFT(n);
        fft->fwd0(OTFFT::complex_vector{spectrum.data()});

        const int bins[] = {0, 1, 2, n / 3, n / 2, n - 2, n - 1};
        for (int k : bins)
        {
            const auto expected = directBin(input, k);
            const auto& got = spectrum[static_cast<std::size_t>(k)];
            CHECK_SMALL(got.Re - expected.Re, 1e-6 * n);
            CHECK_SMALL(got.Im - expected.Im, 1e-6 * n);
        }

        // fwd scales by 1/N, inv does not: together they are the identity
        auto roundtrip = input;
        fft->fwd(OTFFT::complex_vector{roundtrip.data()});
        fft->inv(OTFFT::complex_vector{roundtrip.data()});
        for (std::size_t i = 0; i < roundtrip.size(); i += 97)
        {
            CHECK_SMALL(roundtrip[i].Re - input[i].Re, 1e-8);
            CHECK_SMALL(roundtrip[i].Im - input[i].Im, 1e-8);
        }
    }
}

TEST_SUITE("otfft_prime_test")
{

TEST_CASE("TestComplexPrimeSmall")
{
    // small primes use the mixed radix code, 101 and above Bluestein
    for (int n : {7, 11, 53, 97, 101, 127, 1009})
    {
        checkComplex(n);
    }
}

TEST_CASE("TestComplexLargePrimeFactor")
{
    // a prime factor above the limit next to factors with butterflies
    for (int n : {2 * 101, 8 * 127, 3 * 5 * 251, 4 * 1009})
    {
        checkComplex(n);
    }
}

TEST_CASE("TestComplexPrimeAboveIntOverflow")
{
    // issue #16: from a prime factor of 46,341 the index (i*j)%k overflowed int and the
    // result was wrong (30,011 was right, 50,021 wrong)
    checkComplex(30011);
    checkComplex(50021);
    checkComplex(100003);
}

TEST_CASE("TestComplexPrimeIsFast")
{
    // issue #17: the generic radix was O(N^2), 48 s at N=100,003
    const int n = 100003;
    auto data = makeSignal(n);
    auto fft = OTFFT::Factory::createComplexFFT(n);
    const auto t1 = std::chrono::steady_clock::now();
    fft->fwd0(OTFFT::complex_vector{data.data()});
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    // about 10 ms; the limit is far away from that, so slow machines pass, the O(N^2) code does not
    CHECK_MESSAGE(seconds < 2.0, "fwd0 of a prime length took " << seconds << " s");
}

TEST_CASE("TestComplexSetupChangesPath")
{
    // setup() on an existing object can switch between the mixed radix and the
    // Bluestein path, in both directions
    auto fft = OTFFT::Factory::createComplexFFT();
    for (int n : {1024, 1009, 1000, 4 * 1009, 16})
    {
        fft->setup(n);
        const auto input = makeSignal(n);
        auto spectrum = input;
        fft->fwd0(OTFFT::complex_vector{spectrum.data()});
        for (int k : {0, 1, n - 1})
        {
            const auto expected = directBin(input, k);
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Re - expected.Re, 1e-6 * n);
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Im - expected.Im, 1e-6 * n);
        }
    }
}

TEST_CASE("TestRealPrimeHalfLengthAboveIntOverflow")
{
    // The real FFT of length 2p runs a complex FFT of length p that does not go through
    // createComplexFFT, so it still uses the mixed radix code for a prime p. p = 46,349
    // is above the int overflow limit of issue #16 (about 46,341). Slow (O(p^2), a few
    // seconds): the real FFT is not routed to Bluestein.
    const int n = 2 * 46349;
    const double pi = 3.14159265358979323846;
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        x[static_cast<std::size_t>(i)] = ((i * 7) % 11) - 5.0;
    }
    std::vector<OTFFT::complex_t> y(static_cast<std::size_t>(n));
    auto fft = OTFFT::Factory::createRealFFT(n);
    fft->fwd0(OTFFT::const_double_vector{x.data()}, OTFFT::complex_vector{y.data()});

    const long long ln = n;
    for (int k : {0, 1, 2, n / 4, n / 2 - 1})
    {
        double re = 0;
        double im = 0;
        for (long long j = 0; j < ln; ++j)
        {
            const double angle = -2.0 * pi * static_cast<double>((j * k) % ln) / static_cast<double>(ln);
            re += x[static_cast<std::size_t>(j)] * std::cos(angle);
            im += x[static_cast<std::size_t>(j)] * std::sin(angle);
        }
        CHECK_SMALL(y[static_cast<std::size_t>(k)].Re - re, 1e-6 * n);
        CHECK_SMALL(y[static_cast<std::size_t>(k)].Im - im, 1e-6 * n);
    }
}

}
