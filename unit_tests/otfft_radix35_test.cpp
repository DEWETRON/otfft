// Lengths with the prime factors 3 and 5 (roadmap R1.5): the radix-3 and radix-5
// butterflies use the symmetric form with real constants. Forward and inverse
// transforms are compared with the direct DFT sum in some bins, for the butterflies
// alone, with the other radices around them, and above the OpenMP threshold (2^15).
#include "otfft_dft_check.h"

using OTFFT::test::checkComplexLength;

TEST_SUITE("otfft_radix35_test")
{

TEST_CASE("TestRadix3Only")
{
    for (int n : {3, 9, 27, 81, 243, 729, 2187})
    {
        checkComplexLength(n);
    }
}

TEST_CASE("TestRadix5Only")
{
    for (int n : {5, 25, 125, 625, 3125})
    {
        checkComplexLength(n);
    }
}

TEST_CASE("TestRadix3And5WithOtherFactors")
{
    // with 2, 4 and 8 before and after, and with each other
    for (int n : {6, 12, 24, 48, 96, 10, 20, 40, 80, 160, 15, 45, 75, 225, 375, 480, 960, 1000, 3000, 6000, 3 * 5 * 7 * 8})
    {
        checkComplexLength(n);
    }
}

TEST_CASE("TestRadix3And5AboveOpenMpThreshold")
{
    // N >= 2^15 runs the OpenMP versions of the passes
    for (int n : {3 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3, 5 * 5 * 5 * 5 * 5 * 5 * 5, 3 * 16384, 5 * 8192, 100000, 3375 * 10})
    {
        checkComplexLength(n);
    }
}

}
