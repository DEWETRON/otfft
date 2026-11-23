// Copyright (c) Dewetron 2017
#include "otfft.h"
#include "otfft_test_utils.h"

#include "otfft_test_checks.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{
    std::size_t findPeak(const std::size_t fft_length, const double sample_rate, const double signal_frequency)
    {
        OTFFT::test::SineGenerator generator(sample_rate);
        generator.setAmplitude(64).setFrequency(signal_frequency).generate(fft_length);

        std::vector<OTFFT::complex_t> spectrum(fft_length);
        {
            auto fft = OTFFT::Factory::createRealFFT(static_cast<int>(fft_length));
            OTFFT::double_vector fft_in{generator.data()};
            OTFFT::complex_vector fft_out{spectrum.data()};
            fft->fwd(fft_in, fft_out);
        }

        auto peakPos = std::max_element(
                           std::begin(spectrum),
                           std::begin(spectrum) + fft_length / 2,
                           [] (OTFFT::complex_t x, OTFFT::complex_t y) -> bool
                           {
                               return std::sqrt(OTFFT::norm(x)) < std::sqrt(OTFFT::norm(y));
                           }
                       );

        std::size_t d = std::distance(std::begin(spectrum), peakPos);

        return d;
    }
}

TEST_SUITE("otfft_find_peak_test")
{

TEST_CASE("TestSinePeakDetect1")
{
    const auto peak_position = findPeak(64, 10000, 1000);
    CHECK_EQ(6u, peak_position);
}

TEST_CASE("TestSinePeakDetect2")
{
    const auto peak_position = findPeak(1024, 10000, 50);
    CHECK_EQ(5u, peak_position);
}

TEST_CASE("TestSinePeakDetect3")
{
    const auto peak_position = findPeak(8192, 50000, 4999);
    CHECK_EQ(819u, peak_position);
}

TEST_CASE("TestSinePeakDetect4")
{
    const auto peak_position = findPeak(262144, 1000000, 60000);
    CHECK_EQ(15729u, peak_position);
}

TEST_CASE("TestSinePeakDetect5")
{
    const auto peak_position = findPeak(32, 512, 1);
    CHECK_EQ(0u, peak_position);
}

}
