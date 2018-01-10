// Copyright (c) Dewetron 2017
#include "otfft.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

BOOST_AUTO_TEST_SUITE(otfft_bluestein_test)

BOOST_AUTO_TEST_CASE(TestBluesteinEven)
{
    const auto SIZE = 16;
    std::vector<OTFFT::complex_t> expected{
        {136.0, 136.0},
        {-48.218715937006785, 32.218715937006785},
        {-27.31370849898476, 11.313708498984761},
        {-19.97284610132391, 3.9728461013239116},
        {-16.0, 0.0},
        {-13.34542910335439, -2.6545708966456107},
        {-11.313708498984761, -4.686291501015239},
        {-9.591298939037264, -6.408701060962734},
        {-8.0, -8.0},
        {-6.408701060962734, -9.591298939037264},
        {-4.686291501015239, -11.313708498984761},
        {-2.6545708966456107, -13.34542910335439},
        {0.0, -16.0},
        {3.9728461013239116, -19.97284610132391},
        {11.313708498984761, -27.31370849898476},
        {32.218715937006785, -48.218715937006785}
    };
    std::vector<OTFFT::complex_t> spectrum;

    for (auto n = 1; n <= SIZE; ++n)
    {
        spectrum.emplace_back(n, n);
    }

    auto fft = OTFFT::Factory::createBluesteinFFT(static_cast<int>(SIZE));
    {
        OTFFT::complex_vector spectrum_pointer{spectrum.data()};
        fft->fwd0(spectrum_pointer);
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

    expected.clear();

    for (auto n = 1; n <= SIZE; ++n)
    {
        expected.emplace_back(n, n);
    }

    {
        OTFFT::complex_vector spectrum_pointer{spectrum.data()};
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

BOOST_AUTO_TEST_CASE(TestBluesteinOdd)
{
    const auto SIZE = 13;
    std::vector<OTFFT::complex_t> expected{
        {91.0, 91.0},
        {-32.8715366566478, 19.8715366566477},
        {-18.8847152734699, 5.88471527347}  ,
        {-13.8369832905716, 0.8369832905716},
        {-10.9866260527999, -2.0133739472}   ,
        {-8.9651248401381, -4.0348751598618},
        {-7.2892428909446, -5.7107571090552},
        {-5.7107571090552, -7.2892428909446},
        {-4.0348751598618, -8.9651248401381},
        {-2.0133739472000, -10.9866260527999},
        {0.8369832905716, -13.8369832905716},
        {5.8847152734700, -18.8847152734699},
        {19.8715366566477, -32.8715366566478}
    };
    std::vector<OTFFT::complex_t> spectrum;

    for (auto n = 1; n <= SIZE; ++n)
    {
        spectrum.emplace_back(n, n);
    }

    auto fft = OTFFT::Factory::createBluesteinFFT(static_cast<int>(SIZE));
    {
        OTFFT::complex_vector spectrum_pointer{spectrum.data()};
        fft->fwd0(spectrum_pointer);
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

    expected.clear();

    for (auto n = 1; n <= SIZE; ++n)
    {
        expected.emplace_back(n, n);
    }

    {
        OTFFT::complex_vector spectrum_pointer{spectrum.data()};
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

BOOST_AUTO_TEST_SUITE_END()
