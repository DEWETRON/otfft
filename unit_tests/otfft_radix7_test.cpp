// Lengths with the prime factor 7 (roadmap R1.5, issue #18): the mixed radix code has
// its own radix-7 butterfly. Forward and inverse transforms are compared with the
// direct DFT sum in some bins, for lengths that use the butterfly in the middle and at
// the end of the pass sequence, and for lengths above 2^15 and above the OpenMP threshold (2^19, D30).
#include "otfft_dft_check.h"

using OTFFT::test::checkComplexLength;

TEST_SUITE("otfft_radix7_test")
{

TEST_CASE("TestRadix7Only")
{
    // the butterfly alone, as the last and as the middle pass
    for (int n : {7, 49, 343, 2401})
    {
        checkComplexLength(n);
    }
}

TEST_CASE("TestRadix7WithOtherFactors")
{
    // with 2, 4, 8, 3 and 5 before and after it
    for (int n : {14, 28, 56, 112, 21, 63, 35, 105, 70, 140, 280, 245, 315, 1260, 10080})
    {
        checkComplexLength(n);
    }
}

TEST_CASE("TestRadix7AboveOpenMpThreshold")
{
    // 2^15 <= N < 2^19 is serial since D30; N >= 2^19 runs the OpenMP versions of the passes
    for (int n : {7 * 8192, 7 * 7 * 7 * 7 * 7 * 7, 3 * 5 * 7 * 512, 7 * 131072, 7 * 7 * 7 * 7 * 7 * 7 * 7, 3 * 5 * 7 * 8192})
    {
        checkComplexLength(n);
    }
}

}
