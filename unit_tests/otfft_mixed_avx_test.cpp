// The AVX passes of the mixed radix code (roadmap R1.5, decisions D28 and D29): the radix-3, 5 and 7
// core passes work on two values of q at once when s is even, forward and inverse, serial
// (N < 2^19, D30) and OpenMP (N >= 2^19). Every point of the result is compared with the
// Bluestein transform of the same length, which does not use these passes, for lengths where
// a factor 3, 5 or 7 comes after a factor 2 (so that s is even in that pass, and the pass is
// not the last one). The first pass (s = 1) of radix 8 and radix 4 works on two values of p at
// once; the last p is done by the scalar code when m is odd.
#include "otfft_dft_check.h"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using OTFFT::test::checkComplexLength;
using OTFFT::test::makeSignal;

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

    double maxAbs(const std::vector<OTFFT::complex_t>& v)
    {
        double m = 0;
        for (const auto& z : v) m = std::max(m, std::max(std::fabs(z.Re), std::fabs(z.Im)));
        return m;
    }

    // Largest difference of any point, relative to the largest value.
    double maxRelDiff(const std::vector<OTFFT::complex_t>& a, const std::vector<OTFFT::complex_t>& b)
    {
        double d = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            d = std::max(d, std::max(std::fabs(a[i].Re - b[i].Re), std::fabs(a[i].Im - b[i].Im)));
        }
        return d / std::max(maxAbs(b), 1e-300);
    }

    void checkAgainstBluestein(int n)
    {
        CAPTURE(n);
        const auto input = makeSignal(n);
        auto mixed = OTFFT::Factory::createComplexFFT(n);
        auto reference = OTFFT::Factory::createBluesteinFFT(n);

        auto a = input;
        auto b = input;
        mixed->fwd0(OTFFT::complex_vector{a.data()});
        reference->fwd0(OTFFT::complex_vector{b.data()});
        CHECK(maxRelDiff(a, b) < 1e-12);

        a = input;
        b = input;
        mixed->inv0(OTFFT::complex_vector{a.data()});
        reference->inv0(OTFFT::complex_vector{b.data()});
        CHECK(maxRelDiff(a, b) < 1e-12);
    }
}

TEST_SUITE("otfft_mixed_avx_test")
{

TEST_CASE("TestAvxPassesSerial")
{
    // 2^a times 9, 15, 21, 25, 35, 45, 49, 75, 105: a radix 3, 5 or 7 core pass with an even s
    for (int n : {72, 240, 200, 84, 560, 196, 360, 600, 3360, 1176, 2 * 105, 8 * 9 * 5 * 7, 4 * 25 * 49})
    {
        checkAgainstBluestein(n);
        checkComplexLength(n);
    }
}

TEST_CASE("TestAvxPassesOddS")
{
    // no factor 2: s is odd in every pass, the scalar passes are used
    for (int n : {9, 15, 21, 25, 35, 49, 45, 105, 315, 3 * 5 * 7 * 3, 7 * 7 * 5 * 5})
    {
        checkAgainstBluestein(n);
    }
}

TEST_CASE("TestAvxPassesMediumLengths")
{
    // 2^15 <= N < 2^19: serial since D30, also with several threads
    for (int threads : {1, 4})
    {
        ThreadCount count(threads);
        CAPTURE(threads);
        for (int n : {4096 * 9, 4096 * 15, 2048 * 25, 2048 * 21, 1024 * 35, 1024 * 49, 512 * 105, 2048 * 45, 1024 * 75})
        {
            checkAgainstBluestein(n);
        }
    }
}

TEST_CASE("TestAvxPassesAboveOpenMpThreshold")
{
    // N >= 2^19 runs the OpenMP versions of the passes, with one thread and with several
    for (int threads : {1, 4})
    {
        ThreadCount count(threads);
        CAPTURE(threads);
        for (int n : {65536 * 9, 32768 * 21, 32768 * 25, 16384 * 45, 8192 * 105, 16384 * 75})
        {
            checkAgainstBluestein(n);
        }
    }
}

TEST_CASE("TestAvxFirstPassSerial")
{
    // radix 8 first (a multiple of 8): m = N/8 odd (72, 120, 360, 2520) and even (144, 240, 5040);
    // radix 4 first (4 times an odd number): m = N/4 is always odd (12, 36, 60, 420, 1260, 4 * 25 * 49);
    // radix 2 first (2 times an odd number, scalar)
    for (int n : {24, 72, 120, 360, 2520, 144, 240, 5040, 8 * 7 * 7, 12, 36, 60, 420, 1260, 4 * 25 * 49, 4 * 3 * 3 * 7 * 5, 18, 90, 630})
    {
        checkAgainstBluestein(n);
    }
}

TEST_CASE("TestAvxFirstPassMediumLengths")
{
    // 2^15 <= N < 2^19: serial since D30, m odd and even
    for (int threads : {1, 4})
    {
        ThreadCount count(threads);
        CAPTURE(threads);
        for (int n : {8 * 4725, 3 * 32768, 8 * 5 * 5 * 5 * 5 * 7, 4 * 10125, 4 * 8505, 4 * 3 * 3 * 3 * 5 * 7 * 5 * 7 * 7})
        {
            checkAgainstBluestein(n);
        }
    }
}

TEST_CASE("TestAvxFirstPassAboveOpenMpThreshold")
{
    // N >= 2^19 runs the OpenMP versions: m odd and even, with one thread and with several
    for (int threads : {1, 4})
    {
        ThreadCount count(threads);
        CAPTURE(threads);
        for (int n : {8 * 65625, 3 * 524288, 4 * 212625, 529200})
        {
            checkAgainstBluestein(n);
        }
    }
}

}
