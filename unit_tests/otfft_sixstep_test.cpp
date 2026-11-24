// The OpenMP structure of the SixStep class (roadmap R1.10, decision D20): even lengths use the
// six-step code (one thread below 2^14, a static schedule from 2^14, a guided schedule from
// 2^16), odd lengths the eight-step code (one thread below 2^13, one parallel region with
// sections from 2^13, and from 2^19 one parallel region around the whole step). The table of
// createComplexFFT selects SixStep only for some lengths and thread counts (and with a serial
// table below 2^15), so this test calls the class of each backend directly for every length
// from 2^12 to 2^22, with one, three and eight threads (3 and 8 give an uneven split of the
// eight sections and of the guided chunks). All six transform functions are checked against
// the direct sum.
#include "otfft_class_check.h"

namespace
{
    using OTFFT::test::checkClassLengths;
    using OTFFT::test::ClassTestThreads;

#define OTFFT_SIXSTEP_TEST_BACKEND(NS) \
    checkClassLengths(#NS " SixStep", NS::OTFFT_SixStep::instance, 12, 22);

    void checkAllBackends()
    {
#ifdef OTFFT_WITH_SSE2
        OTFFT_SIXSTEP_TEST_BACKEND(OTFFT_SSE2)
#endif
#ifdef OTFFT_WITH_AVX
        OTFFT_SIXSTEP_TEST_BACKEND(OTFFT_AVX)
#endif
#ifdef OTFFT_WITH_AVX2
        OTFFT_SIXSTEP_TEST_BACKEND(OTFFT_AVX2)
#endif
    }
}

TEST_SUITE("otfft_sixstep_test")
{

TEST_CASE("TestSixStepOneThread")
{
    ClassTestThreads threads(1);
    checkAllBackends();
}

TEST_CASE("TestSixStepThreeThreads")
{
    ClassTestThreads threads(3);
    checkAllBackends();
}

TEST_CASE("TestSixStepEightThreads")
{
    ClassTestThreads threads(8);
    checkAllBackends();
}

}
