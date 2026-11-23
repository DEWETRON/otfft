// Copyright (c) Dewetron 2017
#include "otfft.h"

#include "otfft_test_checks.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

TEST_SUITE("otfft_dct_test")
{

TEST_CASE("TestAlternatingOnes")
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
        CHECK_CLOSE_PCT(testArray[idx], workspace[idx], .1);
    }
}

TEST_CASE("TestConstantInput")
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
        CHECK_CLOSE_PCT(testArray[idx], workspace[idx], .1);
    }
}

TEST_CASE("TestCachedCosines")
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
        CHECK_CLOSE_PCT(testArray[idx], workspace[idx], .1);
    }
}

}
