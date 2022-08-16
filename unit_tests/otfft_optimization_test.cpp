// Copyright (c) Dewetron 2017
#include "otfft.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

namespace
{
    std::ostream& operator<<(std::ostream& out, OTFFT::OptimizationType OPT_TYPE)
    {
        switch (OPT_TYPE)
        {
#ifdef OTFFT_WITH_AVX
        case OTFFT::OptimizationType::OPTIMIZED_FFT_AVX:
            out << "AVX";
            break;
#endif
#ifdef OTFFT_WITH_AVX2
        case OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2:
            out << "AVX2";
            break;
#endif
        case OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2:
            out << "SSE2";
            break;
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, OTFFT::TransformationType TR_TYPE)
    {
        switch (TR_TYPE)
        {
        case OTFFT::TransformationType::TRANSFORM_BLUESTEIN:
            out << "Bluestein";
            break;
        case OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX:
            out << "FftComplex";
            break;
        }
        return out;
    }

    template <OTFFT::TransformationType TR_TYPE, OTFFT::OptimizationType OPT_TYPE>
    void testComplexFFT(const std::size_t SIZE)
    {
        std::random_device rnd_device;
        std::mt19937 mersenne_engine(rnd_device());
        std::uniform_real_distribution<double> dist(-99.0, 99.0);
        auto gen = std::bind(dist, mersenne_engine);

        std::vector<OTFFT::complex_t> expected;
        expected.reserve(SIZE);

        for (auto n = 0; n < SIZE; ++n)
        {
            expected.emplace_back(gen(), gen());
        }

        std::vector<OTFFT::complex_t> spectrum = expected;

        auto fft = TR_TYPE == OTFFT::TransformationType::TRANSFORM_BLUESTEIN ?
                       OTFFT::Factory::createBluesteinFFT(static_cast<int>(SIZE), OPT_TYPE) :
                       OTFFT::Factory::createComplexFFT(static_cast<int>(SIZE), OPT_TYPE);
        {
            OTFFT::complex_vector spectrum_pointer{spectrum.data()};
            fft->fwd0(spectrum_pointer);
            fft->invn(spectrum_pointer);
        }

        for (std::size_t idx{0}; idx < SIZE / 2; ++idx)
        {
            if (std::fabs(expected[idx].Re) < 1e-10)
            {
                BOOST_CHECK_SMALL(spectrum[idx].Re, 1e-8);
            }
            else
            {
                BOOST_CHECK_CLOSE(expected[idx].Re, spectrum[idx].Re, .1);
            }

            if (std::fabs(expected[idx].Im) < 1e-10)
            {
                BOOST_CHECK_SMALL(spectrum[idx].Im, 1e-8);
            }
            else
            {
                BOOST_CHECK_CLOSE(expected[idx].Im, spectrum[idx].Im, .1);
            }
        }
    }

    template <OTFFT::TransformationType TR_TYPE, OTFFT::OptimizationType OPT_TYPE>
    void testComplexSineFFT(const std::size_t SIZE)
    {
        std::vector<OTFFT::complex_t> sine;
        sine.reserve(SIZE);

        const double ampl = 1.5;
        for (auto n = 0; n < SIZE; ++n)
        {
            double t = (2 * M_PI * n) / SIZE;
            sine.emplace_back(ampl * cos(t), ampl * sin(t));
        }
        
        std::vector<OTFFT::complex_t> spectrum = sine;

        auto fft = TR_TYPE == OTFFT::TransformationType::TRANSFORM_BLUESTEIN ?
            OTFFT::Factory::createBluesteinFFT(static_cast<int>(SIZE), OPT_TYPE) :
            OTFFT::Factory::createComplexFFT(static_cast<int>(SIZE), OPT_TYPE);
        {
            OTFFT::complex_vector spectrum_pointer{ spectrum.data() };
            fft->fwdn(spectrum_pointer);
        }

        std::stringstream loc;
        loc << "Error for size " << SIZE << " with " << OPT_TYPE << " and " << TR_TYPE << ": ";

        BOOST_CHECK_MESSAGE(OTFFT::abs(spectrum[0]) < 1e-3, loc.str() << "DC is " << OTFFT::abs(spectrum[0]));

        BOOST_CHECK_MESSAGE(std::abs(OTFFT::abs(spectrum[1]) - ampl) < 1e-3, loc.str() << "amp is " << OTFFT::abs(spectrum[1]));
        BOOST_CHECK_MESSAGE(std::abs(OTFFT::arg(spectrum[1])) < 1e-3, loc.str() << "phase is " << OTFFT::arg(spectrum[1]));

        for (std::size_t n = 2; n < SIZE; ++n)
        {
            //BOOST_CHECK_SMALL(OTFFT::abs(spectrum[n]), 1e-3); // no remaining power
        }
    }

    template <OTFFT::OptimizationType OPT_TYPE>
    void testRealFFT(const std::size_t SIZE)
    {
        std::random_device rnd_device;
        std::mt19937 mersenne_engine(rnd_device());
        std::uniform_real_distribution<double> dist(-99.0, 99.0);
        auto gen = std::bind(dist, mersenne_engine);

        std::vector<double> expected;
        expected.reserve(SIZE);

        for (auto n = 0; n < SIZE; ++n)
        {
            expected.emplace_back(gen());
        }

        std::vector<double> spectrum = expected;

        auto fft = OTFFT::Factory::createRealFFT(static_cast<int>(SIZE), OPT_TYPE);
        {
            std::vector<OTFFT::complex_t> workspace(SIZE);
            OTFFT::double_vector spectrum_pointer{spectrum.data()};
            OTFFT::complex_vector workspace_pointer{workspace.data()};
            fft->fwd0(spectrum_pointer, workspace_pointer);
            fft->invn(workspace_pointer, spectrum_pointer);
        }

        for (std::size_t idx{0}; idx < SIZE / 2; ++idx)
        {
            if (std::fabs(expected[idx]) < 1e-10)
            {
                BOOST_CHECK_SMALL(spectrum[idx], 1e-8);
            }
            else
            {
                BOOST_CHECK_CLOSE(expected[idx], spectrum[idx], .1);
            }
        }
    }

    template <OTFFT::OptimizationType OPT_TYPE>
    void testRealSineFFT(const std::size_t SIZE)
    {
        std::vector<double> sine;
        sine.reserve(SIZE);

        const double ampl = 1.5;
        for (auto n = 0; n < SIZE; ++n)
        {
            double t = (2 * M_PI * n) / SIZE;
            sine.emplace_back(ampl * cos(t));
        }

        std::vector<OTFFT::complex_t> spectrum(sine.size());

        auto fft = OTFFT::Factory::createRealFFT(static_cast<int>(SIZE), OPT_TYPE);
        {
            OTFFT::double_vector input_pointer{ sine.data() };
            OTFFT::complex_vector spectrum_pointer{ spectrum.data() };
            fft->fwdn(input_pointer, spectrum_pointer);
        }

        std::stringstream loc;
        loc << "Error for size " << SIZE << " with " << OPT_TYPE << ": ";

        const auto bin = spectrum[1] + spectrum[SIZE - 1];
        const double ampl_bin = OTFFT::abs(bin);
        const double dc_bin = OTFFT::abs(spectrum[0]);
        BOOST_CHECK_MESSAGE(std::abs(ampl_bin - ampl) < 1e-3, loc.str() << "amp is " << ampl_bin);
        BOOST_CHECK_MESSAGE(dc_bin < 1e-3, loc.str() << "DC is " << dc_bin);

        BOOST_CHECK_CLOSE(OTFFT::abs(bin), ampl, 1e-3); // all power in first bin
        BOOST_CHECK_SMALL(OTFFT::arg(bin), 1e-3); // zero phase in first bin
        for (std::size_t n = 2; n < SIZE / 2; ++n)
        {
            BOOST_CHECK_SMALL(OTFFT::abs(spectrum[n]), 1e-3); // no remaining power
        }
    }

    template <OTFFT::OptimizationType OPT_TYPE>
    void testDCT(const std::size_t SIZE)
    {
        std::random_device rnd_device;
        std::mt19937 mersenne_engine(rnd_device());
        std::uniform_real_distribution<double> dist(-99.0, 99.0);
        auto gen = std::bind(dist, mersenne_engine);

        std::vector<double> expected;

        for (auto n = 0; n < SIZE; ++n)
        {
            expected.emplace_back(gen());
        }

        std::vector<double> spectrum = expected;

        auto fft = OTFFT::Factory::createDCT(static_cast<int>(SIZE), OPT_TYPE);
        {
            OTFFT::double_vector spectrum_pointer{spectrum.data()};
            fft->fwd0(spectrum_pointer);
            fft->invn(spectrum_pointer);
        }

        for (std::size_t idx{0}; idx < SIZE / 2; ++idx)
        {
            if (std::fabs(expected[idx]) < 1e-10)
            {
                BOOST_CHECK_SMALL(spectrum[idx], 1e-8);
            }
            else
            {
                BOOST_CHECK_CLOSE(expected[idx], spectrum[idx], .1);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE(otfft_optimization_test)

BOOST_AUTO_TEST_CASE(TestPowerOfTwoSine)
{
#ifdef OTFFT_WITH_SSE2
    for (int n = 4; n <= (1 << 24); n <<= 2)
    {
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
        testRealSineFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
    }
#endif
}

BOOST_AUTO_TEST_CASE(TestBluestein)
{
    const std::vector<std::size_t> N{
      8, 13, 27, 32, 172, 347, 512, 3247, 4096, 12312, 16384,
      32411, 32768, 53743, 65536, 83476, 131072, 234643, 262144,
      463272, 524288
    };
#ifdef OTFFT_WITH_SSE2
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
    }
#endif
}

BOOST_AUTO_TEST_CASE(TestComplex)
{
    const std::vector<std::size_t> N{
      8, 13, 27, 32, 172, 347, 512, 3247, 4096, 12312, 16384,
      32411, 32768, 53743, 65536, 83476, 131072, 234643, 262144,
      463272, 524288
    };
#ifdef OTFFT_WITH_SSE2
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
        testComplexSineFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
    }
#endif
}

BOOST_AUTO_TEST_CASE(TestReal)
{
    const auto N = {
      8, 32, 64, 128, 256,
      512, 2048, 4096, 16384, 32768, 65536, 131072,
      262144, 524288
    };
#ifdef OTFFT_WITH_SSE2
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
        testRealSineFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
        testRealSineFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
        testRealSineFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
    }
#endif
}

BOOST_AUTO_TEST_CASE(TestRealMixedRadix)
{
    const auto N = {
        2 * 7, 4 * 7, 8 * 7, 
        16 * 5, 16 * 7, 16 * 11,
        32 * 7, 64 * 7, 128 * 7, 256 * 7,
        512 * 7, 1024 * 7, 2048 * 5,
        463272,
    };
#ifdef OTFFT_WITH_SSE2
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
        testRealSineFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
        testRealSineFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
        testRealSineFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
    }
#endif
}

BOOST_AUTO_TEST_CASE(TestDCT)
{
    const std::vector<std::size_t> N{
      8, 32, 512, 4096, 16384, 32768, 65536, 131072,
      262144, 463272, 524288
    };
#ifdef OTFFT_WITH_SSE2
    for (auto n : N)
    {
        testDCT<OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testDCT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testDCT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
    }
#endif
}

BOOST_AUTO_TEST_SUITE_END()
