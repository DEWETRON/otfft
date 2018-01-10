// Copyright (c) Dewetron 2017
#include "otfft.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <random>
#include <vector>

namespace
{
    template <OTFFT::TransformationType TR_TYPE, OTFFT::OptimizationType OPT_TYPE>
    void testComplexFFT(const std::size_t SIZE)
    {
        std::random_device rnd_device;
        std::mt19937 mersenne_engine(rnd_device());
        std::uniform_real_distribution<double> dist(-99.0, 99.0);
        auto gen = std::bind(dist, mersenne_engine);

        std::vector<OTFFT::complex_t> expected;

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

    template <OTFFT::OptimizationType OPT_TYPE>
    void testRealFFT(const std::size_t SIZE)
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
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_BLUESTEIN, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
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
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testComplexFFT<OTFFT::TransformationType::TRANSFORM_FFT_COMPLEX, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
    }
#endif
}

BOOST_AUTO_TEST_CASE(TestReal)
{
    const std::vector<std::size_t> N{
      8, 32, 512, 4096, 16384, 32768, 65536, 131072,
      262144, 463272, 524288
    };
#ifdef OTFFT_WITH_SSE2
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX>(n);
    }
#endif
#ifdef OTFFT_WITH_AVX2
    for (auto n : N)
    {
        testRealFFT<OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2>(n);
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
