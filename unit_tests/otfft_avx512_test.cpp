// The AVX-512 backend (roadmap R2.1, decision D32). It is built only with -DOTFFT_WITH_AVX512=ON, is
// never chosen by OPTIMIZED_FFT_AUTO, and a caller asks for it with OPTIMIZED_FFT_AVX512. As for the other
// backends, a request for a backend that the CPU cannot run throws std::domain_error.
//
// On a CPU with AVX512F, AVX512DQ and AVX512VL the backend must give the same results as the default
// one; on any other CPU the request must throw and the default plans must still work. Only one of the
// two halves runs on a given machine: the first needs AVX-512 hardware.
#include "otfft.h"

#ifdef OTFFT_WITH_AVX512

#include "otfft_test_checks.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{
    // true if the CPU can run the backend; with a compiler without the builtin it is not known
    // (-1), then the test accepts both outcomes
    int cpuRunsAvx512()
    {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        return (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
                __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
                   ? 1 : 0;
#else
        return -1;
#endif
    }

    const OTFFT::OptimizationType kAvx512 = OTFFT::OptimizationType::OPTIMIZED_FFT_AVX512;

    std::vector<OTFFT::complex_t> makeSignal(int n)
    {
        std::vector<OTFFT::complex_t> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = OTFFT::complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        return x;
    }

    // Runs `make` with the AVX-512 request and compares with the default plan; returns false if
    // the request was refused with std::domain_error.
    template <class Make>
    bool compareComplex(int n, Make make)
    {
        const auto input = makeSignal(n);
        auto reference = input;
        make(n, OTFFT::OptimizationType::OPTIMIZED_FFT_AUTO)->fwd0(OTFFT::complex_vector{reference.data()});
        OTFFT::ComplexFFTPtr plan;
        try
        {
            plan = make(n, kAvx512);
        }
        catch (const std::domain_error&)
        {
            return false;
        }
        auto result = input;
        plan->fwd0(OTFFT::complex_vector{result.data()});
        double d = 0;
        double m = 0;
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            d = std::max(d, std::max(std::fabs(result[i].Re - reference[i].Re), std::fabs(result[i].Im - reference[i].Im)));
            m = std::max(m, std::max(std::fabs(reference[i].Re), std::fabs(reference[i].Im)));
        }
        CHECK(d <= 1e-12 * std::max(m, 1.0) * n);
        return true;
    }

    void checkOutcome(bool ran)
    {
        const int expected = cpuRunsAvx512();
        if (expected == 1) CHECK(ran);
        if (expected == 0) CHECK_FALSE(ran);
    }
}

TEST_SUITE("otfft_avx512_test")
{

TEST_CASE("TestAutoStillWorks")
{
    // OPTIMIZED_FFT_AUTO never chooses the AVX-512 backend and is not disturbed by its presence
    auto fft = OTFFT::Factory::createComplexFFT(1000);
    auto x = makeSignal(1000);
    fft->fwd0(OTFFT::complex_vector{x.data()});
    CHECK(std::isfinite(x[1].Re));
    CHECK_NOTHROW(OTFFT::Factory::createRealFFT(1024));
    CHECK_NOTHROW(OTFFT::Factory::createDCT(1024));
    CHECK_NOTHROW(OTFFT::Factory::createBluesteinFFT(1009));
}

TEST_CASE("TestComplexFftOnRequest")
{
    // power of two, mixed radix (factor 2 before the odd factors), Bluestein path
    for (int n : {64, 4096, 1000, 15360, 1009})
    {
        CAPTURE(n);
        checkOutcome(compareComplex(n, [](int len, OTFFT::OptimizationType t) { return OTFFT::Factory::createComplexFFT(len, t); }));
    }
}

TEST_CASE("TestBluesteinOnRequest")
{
    for (int n : {1000, 1009})
    {
        CAPTURE(n);
        checkOutcome(compareComplex(n, [](int len, OTFFT::OptimizationType t) { return OTFFT::Factory::createBluesteinFFT(len, t); }));
    }
}

TEST_CASE("TestRealFftAndDctOnRequest")
{
    const int n = 2048;
    std::vector<double> input(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) input[static_cast<std::size_t>(i)] = ((i * 7) % 11) - 5.0;

    auto referenceDct = input;
    OTFFT::Factory::createDCT(n)->fwd0(referenceDct.data());
    std::vector<OTFFT::complex_t> referenceSpectrum(static_cast<std::size_t>(n) + 2);
    OTFFT::Factory::createRealFFT(n)->fwd0(input.data(), OTFFT::complex_vector{referenceSpectrum.data()});

    bool ranDct = true;
    try
    {
        auto dct = OTFFT::Factory::createDCT(n, kAvx512);
        auto x = input;
        dct->fwd0(x.data());
        double d = 0;
        for (int i = 0; i < n; ++i) d = std::max(d, std::fabs(x[static_cast<std::size_t>(i)] - referenceDct[static_cast<std::size_t>(i)]));
        CHECK(d < 1e-9);
    }
    catch (const std::domain_error&)
    {
        ranDct = false;
    }
    checkOutcome(ranDct);

    bool ranReal = true;
    try
    {
        auto rfft = OTFFT::Factory::createRealFFT(n, kAvx512);
        std::vector<OTFFT::complex_t> spectrum(static_cast<std::size_t>(n) + 2);
        rfft->fwd0(input.data(), OTFFT::complex_vector{spectrum.data()});
        double d = 0;
        for (int k = 0; k <= n / 2; ++k)
        {
            d = std::max(d, std::max(std::fabs(spectrum[static_cast<std::size_t>(k)].Re - referenceSpectrum[static_cast<std::size_t>(k)].Re),
                                     std::fabs(spectrum[static_cast<std::size_t>(k)].Im - referenceSpectrum[static_cast<std::size_t>(k)].Im)));
        }
        CHECK(d < 1e-8);
    }
    catch (const std::domain_error&)
    {
        ranReal = false;
    }
    checkOutcome(ranReal);
}

}

#endif // OTFFT_WITH_AVX512
