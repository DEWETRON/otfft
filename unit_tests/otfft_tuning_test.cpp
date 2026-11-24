// Run-time tuning file for the complex FFT algorithm tables (roadmap R1.8, decision D22,
// spec FAC-9 to FAC-13).
#include "otfft.h"
#include "otfft_tuning.h"

#include "otfft_test_checks.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    using OTFFT::Tuning::BACKEND_AVX;
    using OTFFT::Tuning::BACKEND_AVX2;
    using OTFFT::Tuning::BACKEND_SSE2;
    using OTFFT::Tuning::lookup;

    const char* const kFile = "otfft_tuning_test_file.txt";

    // Resets the tuning before and after a test case, so cases do not depend on each other.
    struct TuningGuard
    {
        TuningGuard() { OTFFT::resetTuning(); }
        ~TuningGuard()
        {
            OTFFT::resetTuning();
            std::remove(kFile);
        }
    };

    void writeFile(const std::string& text)
    {
        std::ofstream out(kFile);
        out << text;
    }

    bool load(const std::string& text, std::string& error)
    {
        writeFile(text);
        return OTFFT::loadTuningFile(kFile, error);
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

    // Forward spectrum bins of createComplexFFT(n) against the direct sum.
    void checkLength(int log_n)
    {
        const int n = 1 << log_n;
        const double pi = 3.14159265358979323846;
        std::vector<OTFFT::complex_t> input(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            input[static_cast<std::size_t>(i)] = OTFFT::complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        }
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

    // A file that sets algorithm `name` at length 2^log_n for all backends and both tables.
    std::string everywhere(int log_n, const char* name)
    {
        std::string text = "otfft-tuning 1\n";
        for (const char* backend : {"sse2", "avx", "avx2"})
        {
            for (const char* table : {"serial", "threaded"})
            {
                text += std::string(backend) + " " + table + " " + std::to_string(log_n) + " " + name + "\n";
            }
        }
        return text;
    }
}

TEST_SUITE("otfft_tuning_test")
{

TEST_CASE("TestNothingLoadedByDefault")
{
    TuningGuard guard;
    for (int log_n = 1; log_n <= 24; ++log_n)
    {
        CHECK(lookup(BACKEND_AVX2, false, log_n) == 0);
        CHECK(lookup(BACKEND_AVX2, true, log_n) == 0);
    }
}

TEST_CASE("TestLoadSetsListedEntriesOnly")
{
    TuningGuard guard;
    std::string error;
    const std::string text =
        "# written by hand\n"
        "otfft-tuning 1   # header\n"
        "\n"
        "avx2 serial 17 DIF4\n"
        "avx2 threaded 17 SixStep   # a comment\n"
        "avx serial 18 DIF16\n"
        "sse2 serial 12 DIT16\n";
    REQUIRE(load(text, error));
    CHECK(error.empty());
    CHECK(lookup(BACKEND_AVX2, false, 17) == 1);
    CHECK(lookup(BACKEND_AVX2, true, 17) == 7);
    CHECK(lookup(BACKEND_AVX, false, 18) == 5);
    CHECK(lookup(BACKEND_SSE2, false, 12) == 6);
    // everything not listed keeps the built-in choice
    CHECK(lookup(BACKEND_AVX2, false, 18) == 0);
    CHECK(lookup(BACKEND_AVX, false, 17) == 0);
    CHECK(lookup(BACKEND_SSE2, true, 12) == 0);
}

TEST_CASE("TestAllAlgorithmNames")
{
    TuningGuard guard;
    std::string error;
    const char* const names[] = {"DIF4", "DIT4", "DIF8", "DIT8", "DIF16", "DIT16", "SixStep"};
    std::string text = "otfft-tuning 1\n";
    for (int i = 0; i < 7; ++i)
    {
        text += std::string("avx2 serial ") + std::to_string(i + 1) + " " + names[i] + "\n";
    }
    REQUIRE(load(text, error));
    for (int i = 0; i < 7; ++i)
    {
        CHECK(lookup(BACKEND_AVX2, false, i + 1) == i + 1);
    }
}

TEST_CASE("TestLastEntryWins")
{
    TuningGuard guard;
    std::string error;
    REQUIRE(load("otfft-tuning 1\navx2 serial 14 DIF4\navx2 serial 14 DIF16\n", error));
    CHECK(lookup(BACKEND_AVX2, false, 14) == 5);
}

TEST_CASE("TestLoadReplacesEarlierFile")
{
    TuningGuard guard;
    std::string error;
    REQUIRE(load("otfft-tuning 1\navx2 serial 14 DIF4\n", error));
    REQUIRE(load("otfft-tuning 1\navx2 serial 15 DIF8\n", error));
    CHECK(lookup(BACKEND_AVX2, false, 14) == 0);
    CHECK(lookup(BACKEND_AVX2, false, 15) == 3);
}

TEST_CASE("TestResetDropsTheFile")
{
    TuningGuard guard;
    std::string error;
    REQUIRE(load("otfft-tuning 1\navx2 serial 14 DIF4\n", error));
    OTFFT::resetTuning();
    CHECK(lookup(BACKEND_AVX2, false, 14) == 0);
}

TEST_CASE("TestBadFilesAreRejectedAndChangeNothing")
{
    TuningGuard guard;
    std::string error;
    REQUIRE(load("otfft-tuning 1\navx2 serial 14 DIF4\n", error));

    const char* const bad[] = {
        "",                                                  // empty
        "# only a comment\n",                                // no header
        "avx2 serial 14 DIF4\n",                             // entry before the header
        "otfft-tuning 2\n",                                  // unknown version
        "otfft-tuning 1\nneon serial 14 DIF4\n",             // unknown backend
        "otfft-tuning 1\navx2 parallel 14 DIF4\n",           // unknown table
        "otfft-tuning 1\navx2 serial 0 DIF4\n",              // length below 2^1
        "otfft-tuning 1\navx2 serial 25 DIF4\n",             // length above 2^24
        "otfft-tuning 1\navx2 serial x DIF4\n",              // not a number
        "otfft-tuning 1\navx2 serial 14 DIF32\n",            // unknown algorithm
        "otfft-tuning 1\navx2 serial 14 8\n",                // numbers are not algorithm names
        "otfft-tuning 1\navx2 serial 14\n",                  // too few words
        "otfft-tuning 1\navx2 serial 14 DIF4 DIF8\n",        // too many words
        "otfft-tuning 1\navx2 serial 15 DIF8\navx2 serial 99 DIF8\n",  // good line, then a bad one
    };
    for (const char* text : bad)
    {
        CAPTURE(text);
        error.clear();
        CHECK_FALSE(load(text, error));
        CHECK_FALSE(error.empty());
        // the earlier, valid tuning is untouched
        CHECK(lookup(BACKEND_AVX2, false, 14) == 1);
        CHECK(lookup(BACKEND_AVX2, false, 15) == 0);
    }
}

TEST_CASE("TestErrorNamesTheLine")
{
    TuningGuard guard;
    std::string error;
    CHECK_FALSE(load("otfft-tuning 1\navx2 serial 14 DIF4\navx2 serial 14 nope\n", error));
    CHECK(error.find("line 3") != std::string::npos);
}

TEST_CASE("TestMissingFile")
{
    TuningGuard guard;
    std::string error;
    CHECK_FALSE(OTFFT::loadTuningFile("otfft_tuning_test_no_such_file.txt", error));
    CHECK_FALSE(error.empty());
}

TEST_CASE("TestEveryAlgorithmGivesTheRightSpectrumOneThread")
{
    TuningGuard guard;
    ThreadCount threads(1);
    std::string error;
    for (const char* name : {"DIF4", "DIT4", "DIF8", "DIT8", "DIF16", "DIT16", "SixStep"})
    {
        CAPTURE(name);
        REQUIRE(load(everywhere(13, name), error));
        checkLength(13);
    }
}

TEST_CASE("TestEveryAlgorithmGivesTheRightSpectrumSeveralThreads")
{
    TuningGuard guard;
    ThreadCount threads(4);
    std::string error;
    for (const char* name : {"DIF4", "DIT4", "DIF8", "DIT8", "DIF16", "DIT16", "SixStep"})
    {
        CAPTURE(name);
        REQUIRE(load(everywhere(16, name), error));
        checkLength(16);
    }
}

TEST_CASE("TestPlanKeepsItsAlgorithmWhenTheFileChanges")
{
    TuningGuard guard;
    ThreadCount threads(1);
    std::string error;
    REQUIRE(load(everywhere(12, "DIF4"), error));
    auto fft = OTFFT::Factory::createComplexFFT(1 << 12);
    REQUIRE(load(everywhere(12, "DIT16"), error));
    // the plan made before keeps working
    std::vector<OTFFT::complex_t> x(1 << 12, OTFFT::complex_t(1.0, 0.0));
    fft->fwd0(OTFFT::complex_vector{x.data()});
    CHECK_SMALL(x[0].Re - (1 << 12), 1e-9);
    CHECK_SMALL(x[1].Re, 1e-9);
}

}
