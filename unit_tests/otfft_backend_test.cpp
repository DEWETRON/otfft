// What each backend is compiled for (roadmap R1.7 and R2.1, decisions D21 and D32). The backends share all of
// their code; they differ only in the compiler flags. GCC and Clang do not enable FMA with
// -mavx2, so the AVX2 backend was once built without it and measured no faster than AVX.
// This test fails if a build flag is lost.
#include "otfft.h"

#include "otfft_test_checks.h"

#ifdef OTFFT_WITH_SSE2
namespace OTFFT_SSE2
{
    bool compiledWithAvx();
    bool compiledWithAvx2();
    bool compiledWithAvx512();
    bool compiledWithFma();
}
#endif
#ifdef OTFFT_WITH_AVX
namespace OTFFT_AVX
{
    bool compiledWithAvx();
    bool compiledWithAvx2();
    bool compiledWithAvx512();
    bool compiledWithFma();
}
#endif
#ifdef OTFFT_WITH_AVX2
namespace OTFFT_AVX2
{
    bool compiledWithAvx();
    bool compiledWithAvx2();
    bool compiledWithAvx512();
    bool compiledWithFma();
}
#endif
#ifdef OTFFT_WITH_AVX512
namespace OTFFT_AVX512
{
    bool compiledWithAvx();
    bool compiledWithAvx2();
    bool compiledWithAvx512();
    bool compiledWithFma();
}
#endif

TEST_SUITE("otfft_backend_test")
{

#ifdef OTFFT_WITH_SSE2
TEST_CASE("TestSse2BackendUsesNoAvx")
{
    CHECK_FALSE(OTFFT_SSE2::compiledWithAvx());
    CHECK_FALSE(OTFFT_SSE2::compiledWithAvx2());
    CHECK_FALSE(OTFFT_SSE2::compiledWithAvx512());
    CHECK_FALSE(OTFFT_SSE2::compiledWithFma());
}
#endif

#ifdef OTFFT_WITH_AVX
TEST_CASE("TestAvxBackendUsesAvxWithoutFma")
{
    CHECK(OTFFT_AVX::compiledWithAvx());
    CHECK_FALSE(OTFFT_AVX::compiledWithAvx2());
    CHECK_FALSE(OTFFT_AVX::compiledWithAvx512());
    CHECK_FALSE(OTFFT_AVX::compiledWithFma());
}
#endif

#ifdef OTFFT_WITH_AVX2
TEST_CASE("TestAvx2BackendUsesAvx2AndFma")
{
    CHECK(OTFFT_AVX2::compiledWithAvx());
    CHECK(OTFFT_AVX2::compiledWithAvx2());
    CHECK_FALSE(OTFFT_AVX2::compiledWithAvx512());
    CHECK(OTFFT_AVX2::compiledWithFma());
}
#endif

#ifdef OTFFT_WITH_AVX512
TEST_CASE("TestAvx512BackendUsesAvx512WithAvx2AndFma")
{
    CHECK(OTFFT_AVX512::compiledWithAvx());
    CHECK(OTFFT_AVX512::compiledWithAvx2());
    CHECK(OTFFT_AVX512::compiledWithAvx512());
    CHECK(OTFFT_AVX512::compiledWithFma());
}
#endif

}
