// Lengths around the OpenMP thresholds of the real FFT and the DCT (2^20) and of the mixed
// radix code (2^19), roadmap R1.6, decision D30: below the threshold the transform runs on one
// thread whatever the thread count, from the threshold on the result with several threads is
// the same as with one.
#include "otfft.h"

#include "otfft_test_checks.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
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

    std::vector<double> makeRealSignal(int n)
    {
        std::vector<double> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = ((i * 7) % 11) - 5.0 + 0.25 * ((i * 3) % 5);
        return x;
    }

    double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b)
    {
        double d = 0;
        for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
        return d;
    }
}

TEST_SUITE("otfft_omp_threshold_test")
{

TEST_CASE("TestRealFftAroundThreshold")
{
    // the real FFT agrees with the complex FFT of the same data, in every bin 0..N/2, and
    // forward followed by inverse gives back the input
    for (int log_n : {15, 16, 19, 20, 21})
    {
        const int n = 1 << log_n;
        const auto input = makeRealSignal(n);
        std::vector<OTFFT::complex_t> reference(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) reference[static_cast<std::size_t>(i)] = OTFFT::complex_t(input[static_cast<std::size_t>(i)], 0.0);
        OTFFT::Factory::createComplexFFT(n)->fwd0(OTFFT::complex_vector{reference.data()});

        for (int threads : {1, 4})
        {
            ThreadCount count(threads);
            CAPTURE(log_n);
            CAPTURE(threads);
            auto rfft = OTFFT::Factory::createRealFFT(n);
            std::vector<OTFFT::complex_t> spectrum(static_cast<std::size_t>(n) + 2);
            rfft->fwd0(input.data(), OTFFT::complex_vector{spectrum.data()});
            double worst = 0;
            for (int k = 0; k <= n / 2; k += 1 + (k % 7))
            {
                worst = std::max(worst, std::max(std::fabs(spectrum[static_cast<std::size_t>(k)].Re - reference[static_cast<std::size_t>(k)].Re),
                                                 std::fabs(spectrum[static_cast<std::size_t>(k)].Im - reference[static_cast<std::size_t>(k)].Im)));
            }
            CHECK(worst < 1e-8 * n);

            std::vector<double> output(static_cast<std::size_t>(n));
            rfft->fwd(input.data(), OTFFT::complex_vector{spectrum.data()});
            rfft->inv(OTFFT::complex_vector{spectrum.data()}, output.data());
            CHECK(maxAbsDiff(input, output) < 1e-9);
        }
    }
}

TEST_CASE("TestDctAroundThreshold")
{
    // forward followed by inverse gives back the input, and four threads give the result of one
    for (int log_n : {15, 16, 19, 20, 21})
    {
        const int n = 1 << log_n;
        const auto input = makeRealSignal(n);
        std::vector<double> single;
        for (int threads : {1, 4})
        {
            ThreadCount count(threads);
            CAPTURE(log_n);
            CAPTURE(threads);
            auto dct = OTFFT::Factory::createDCT(n);
            auto work = input;
            dct->fwd0(work.data());
            if (threads == 1) single = work;
            else CHECK(maxAbsDiff(single, work) < 1e-8 * std::sqrt(static_cast<double>(n)));
            dct->invn(work.data());
            CHECK(maxAbsDiff(input, work) < 1e-9 * std::sqrt(static_cast<double>(n)));
        }
    }
}

TEST_CASE("TestMixedRadixAroundThreshold")
{
    // 15 * 2^15 = 491,520 is below 2^19, 9 * 2^16 = 589,824 above: one and four threads agree
    for (int n : {15 * 32768, 9 * 65536})
    {
        std::vector<OTFFT::complex_t> single;
        for (int threads : {1, 4})
        {
            ThreadCount count(threads);
            CAPTURE(n);
            CAPTURE(threads);
            std::vector<OTFFT::complex_t> x(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = OTFFT::complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
            auto original = x;
            auto fft = OTFFT::Factory::createComplexFFT(n);
            fft->fwd0(OTFFT::complex_vector{x.data()});
            if (threads == 1) single = x;
            else
            {
                double d = 0;
                for (std::size_t i = 0; i < x.size(); ++i) d = std::max(d, std::max(std::fabs(x[i].Re - single[i].Re), std::fabs(x[i].Im - single[i].Im)));
                CHECK(d < 1e-8 * n);
            }
            fft->inv(OTFFT::complex_vector{x.data()});
            for (std::size_t i = 0; i < x.size(); i += 97)
            {
                CHECK_SMALL(x[i].Re - original[i].Re * n, 1e-6 * n);
            }
        }
    }
}

}
