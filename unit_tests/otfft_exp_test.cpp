// The exp kernels of the radix-8 and radix-16 classes (roadmap R1.9, decision D19): from
// N = 2^18 the classes DIF8, DIT8, DIF16 and DIT16 compute the twiddle factors in registers
// instead of reading a weight table. The table of createComplexFFT picks one class per length,
// so this test calls each class of each backend directly. It checks all six transform
// functions (scales 1/N, 1, 1/sqrt(N)) against the direct sum, for lengths 2^18 to 2^21 (all
// remainders of log2(N) modulo 3 and 4), with one thread and with four.
#include "otfft_class_check.h"

namespace
{
    using OTFFT::test::checkClassLengths;
    using OTFFT::test::ClassTestThreads;

#define OTFFT_EXP_TEST_BACKEND(NS)                                                           \
    checkClassLengths(#NS " DIF8", NS::OTFFT_AVXDIF8::instance, 18, 21);                     \
    checkClassLengths(#NS " DIT8", NS::OTFFT_AVXDIT8::instance, 18, 21);                     \
    checkClassLengths(#NS " DIF16", NS::OTFFT_AVXDIF16::instance, 18, 21);                   \
    checkClassLengths(#NS " DIT16", NS::OTFFT_AVXDIT16::instance, 18, 21);

    void checkAllBackends()
    {
#ifdef OTFFT_WITH_SSE2
        OTFFT_EXP_TEST_BACKEND(OTFFT_SSE2)
#endif
#ifdef OTFFT_WITH_AVX
        OTFFT_EXP_TEST_BACKEND(OTFFT_AVX)
#endif
#ifdef OTFFT_WITH_AVX2
        OTFFT_EXP_TEST_BACKEND(OTFFT_AVX2)
#endif
    }
}

TEST_SUITE("otfft_exp_test")
{

TEST_CASE("TestExpKernelsOneThread")
{
    ClassTestThreads threads(1);
    checkAllBackends();
}

TEST_CASE("TestExpKernelsFourThreads")
{
    ClassTestThreads threads(4);
    checkAllBackends();
}

}
