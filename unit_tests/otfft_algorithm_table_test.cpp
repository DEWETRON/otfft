// The per-length algorithm choice of the complex FFT (roadmap R1.8, decision D14):
// every power-of-two length gives the right spectrum with one thread and with
// several, and a plan made with one thread stays correct when the thread count
// changes afterwards.
#include "otfft.h"

#include "otfft_test_checks.h"

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

    // Forward spectrum bins of createComplexFFT(n) against the direct sum (O(n) per bin).
    void checkLength(int log_n)
    {
        const int n = 1 << log_n;
        const double pi = 3.14159265358979323846;
        const auto input = makeSignal(n);
        auto spectrum = input;
        auto fft = OTFFT::Factory::createComplexFFT(n);
        fft->fwd0(OTFFT::complex_vector{spectrum.data()});

        const int bins[] = {0, 1, n / 3, n - 1};
        for (int k : bins)
        {
            double re = 0;
            double im = 0;
            for (long long j = 0; j < n; ++j)
            {
                const double angle = -2.0 * pi * static_cast<double>((j * k) & (n - 1)) / static_cast<double>(n);
                const auto& v = input[static_cast<std::size_t>(j)];
                re += v.Re * std::cos(angle) - v.Im * std::sin(angle);
                im += v.Re * std::sin(angle) + v.Im * std::cos(angle);
            }
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Re - re, 1e-9 * n);
            CHECK_SMALL(spectrum[static_cast<std::size_t>(k)].Im - im, 1e-9 * n);
        }
    }

    struct ThreadCount
    {
#ifdef _OPENMP
        int saved = omp_get_max_threads();
        explicit ThreadCount(int n) { omp_set_num_threads(n); }
        ~ThreadCount() { omp_set_num_threads(saved); }
#else
        explicit ThreadCount(int) {}
#endif
    };
}

TEST_SUITE("otfft_algorithm_table_test")
{

TEST_CASE("TestPowerOfTwoOneThread")
{
    ThreadCount threads(1);
    for (int log_n = 1; log_n <= 22; ++log_n)
    {
        CAPTURE(log_n);
        checkLength(log_n);
    }
}

TEST_CASE("TestPowerOfTwoSeveralThreads")
{
    ThreadCount threads(4);
    for (int log_n = 1; log_n <= 22; ++log_n)
    {
        CAPTURE(log_n);
        checkLength(log_n);
    }
}

TEST_CASE("TestPlanSurvivesThreadCountChange")
{
    // the table is chosen in setup(), from the thread count at that time
    for (int log_n : {12, 16, 18, 20})
    {
        const int n = 1 << log_n;
        const auto input = makeSignal(n);
        OTFFT::ComplexFFTPtr fft;
        {
            ThreadCount one(1);
            fft = OTFFT::Factory::createComplexFFT(n);
        }
        ThreadCount four(4);
        auto spectrum = input;
        fft->fwd(OTFFT::complex_vector{spectrum.data()});
        fft->inv(OTFFT::complex_vector{spectrum.data()});
        for (std::size_t i = 0; i < spectrum.size(); i += 101)
        {
            CHECK_SMALL(spectrum[i].Re - input[i].Re, 1e-8);
            CHECK_SMALL(spectrum[i].Im - input[i].Im, 1e-8);
        }
    }
}

}
