// The work buffer of the complex FFT is moved inside its allocation so that its distance to the
// input is a fixed value modulo 4096 bytes (roadmap R1.5, decision D31). The result must not depend
// on where the input is: the input is placed at every multiple of 32 bytes inside a page, the
// largest shift of the work buffer (4064 bytes) is among them.
#include "otfft.h"

#include "otfft_test_checks.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{
    // A block of memory with room for the input at any offset 0, 32, ..., 4064 bytes.
    struct Block
    {
        explicit Block(std::size_t bytes) : p(OTFFT::simd_malloc(bytes + 4096)) {}
        ~Block() { OTFFT::simd_free(p); }
        char* at(std::size_t offset) const { return static_cast<char*>(p) + offset; }
        void* p;
    };

    std::vector<OTFFT::complex_t> makeSignal(int n)
    {
        std::vector<OTFFT::complex_t> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = OTFFT::complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        return x;
    }

    void checkLength(int n)
    {
        CAPTURE(n);
        const auto input = makeSignal(n);
        auto fft = OTFFT::Factory::createComplexFFT(n);

        auto reference = input;
        fft->fwd0(OTFFT::complex_vector{reference.data()});

        Block block(static_cast<std::size_t>(n) * sizeof(OTFFT::complex_t));
        for (std::size_t offset = 0; offset < 4096; offset += 32)
        {
            CAPTURE(offset);
            auto* x = reinterpret_cast<OTFFT::complex_t*>(block.at(offset));
            std::copy(input.begin(), input.end(), x);
            fft->fwd0(OTFFT::complex_vector{x});
            double d = 0;
            for (int i = 0; i < n; ++i)
            {
                d = std::max(d, std::max(std::fabs(x[i].Re - reference[static_cast<std::size_t>(i)].Re),
                                         std::fabs(x[i].Im - reference[static_cast<std::size_t>(i)].Im)));
            }
            CHECK(d < 1e-9);

            // and back: inverse after forward gives n times the input
            fft->inv0(OTFFT::complex_vector{x});
            double e = 0;
            for (int i = 0; i < n; i += 13)
            {
                e = std::max(e, std::fabs(x[i].Re - n * input[static_cast<std::size_t>(i)].Re));
            }
            CHECK(e < 1e-6 * n);
        }
    }
}

TEST_SUITE("otfft_placement_test")
{

TEST_CASE("TestPowerOfTwoAtEveryPageOffset")
{
    for (int n : {64, 1024, 4096})
    {
        checkLength(n);
    }
}

TEST_CASE("TestMixedRadixAtEveryPageOffset")
{
    // a factor 2 before the odd factors (AVX passes), odd lengths, a first pass of radix 4 and 8
    for (int n : {96, 1000, 1260, 3840, 2520})
    {
        checkLength(n);
    }
}

TEST_CASE("TestBluesteinPathAtEveryPageOffset")
{
    // a prime factor above 100: the plan uses Bluestein, which has its own buffers
    for (int n : {1009, 2 * 1009})
    {
        checkLength(n);
    }
}

TEST_CASE("TestRealFftAndDctAtEveryPageOffset")
{
    // the real FFT and the DCT run a complex FFT on their own buffers
    const int n = 2048;
    std::vector<double> input(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) input[static_cast<std::size_t>(i)] = ((i * 7) % 11) - 5.0;
    auto dct = OTFFT::Factory::createDCT(n);
    auto reference = input;
    dct->fwd0(reference.data());

    Block block(static_cast<std::size_t>(n) * sizeof(double));
    for (std::size_t offset = 0; offset < 4096; offset += 32)
    {
        CAPTURE(offset);
        auto* x = reinterpret_cast<double*>(block.at(offset));
        std::copy(input.begin(), input.end(), x);
        dct->fwd0(x);
        double d = 0;
        for (int i = 0; i < n; ++i) d = std::max(d, std::fabs(x[i] - reference[static_cast<std::size_t>(i)]));
        CHECK(d < 1e-9);
    }
}

}
