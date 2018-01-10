// Copyright (c) Dewetron 2017
#include "otfft.h"

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

BOOST_AUTO_TEST_SUITE(otfft_dct_test)

BOOST_AUTO_TEST_CASE(TestAlternatingOnes)
{
    const std::size_t SIZE = 4;
    const double testArray[SIZE] = {1.0, -1.0, 1.0, -1.0};

    std::vector<double> workspace(testArray, testArray + SIZE);
    {
        auto dct = OTFFT::Factory::createDCT(static_cast<int>(SIZE));
        dct->fwd0(workspace.data());
        dct->invn(workspace.data());
    }

    for (std::size_t idx{0}; idx < SIZE; ++idx)
    {
        BOOST_CHECK_CLOSE(testArray[idx], workspace[idx], .1);
    }
}

BOOST_AUTO_TEST_CASE(TestConstantInput)
{
    const std::size_t SIZE = 4;
    const double testArray[SIZE] = {1.0, 1.0, 1.0, 1.0};

    std::vector<double> workspace(testArray, testArray + SIZE);
    {
        auto dct = OTFFT::Factory::createDCT(static_cast<int>(SIZE));
        dct->fwd0(workspace.data());
        dct->invn(workspace.data());
    }

    for (std::size_t idx{0}; idx < SIZE; ++idx)
    {
        BOOST_CHECK_CLOSE(testArray[idx], workspace[idx], .1);
    }
}

BOOST_AUTO_TEST_CASE(TestCachedCosines)
{
    const std::size_t SIZE = 4;
    const double testArray[SIZE] = {1.0, 1.0, 1.0, 1.0};

    std::vector<double> workspace(testArray, testArray + SIZE);
    {
        auto dct = OTFFT::Factory::createDCT(static_cast<int>(SIZE));
        dct->fwd0(workspace.data());
        dct->fwd0(workspace.data());
        dct->invn(workspace.data());
        dct->invn(workspace.data());
    }

    for (std::size_t idx{0}; idx < SIZE; ++idx)
    {
        BOOST_CHECK_CLOSE(testArray[idx], workspace[idx], .1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
