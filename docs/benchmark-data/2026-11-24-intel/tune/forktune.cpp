// Times the fork's own per-algorithm classes (instance() factories in the backend libs).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>
#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"
namespace OTFFT_NAMESPACE {
#include "otfft_if.h"
}

#define DECL(ns) namespace OTFFT_NAMESPACE { \
    namespace OTFFT_AVXDIF4 { extern std::unique_ptr<FFT_IF> instance(); } \
    namespace OTFFT_AVXDIT4 { extern std::unique_ptr<FFT_IF> instance(); } \
    namespace OTFFT_AVXDIF8 { extern std::unique_ptr<FFT_IF> instance(); } \
    namespace OTFFT_AVXDIT8 { extern std::unique_ptr<FFT_IF> instance(); } \
    namespace OTFFT_AVXDIF16 { extern std::unique_ptr<FFT_IF> instance(); } \
    namespace OTFFT_AVXDIT16 { extern std::unique_ptr<FFT_IF> instance(); } \
    namespace OTFFT_SixStep { extern std::unique_ptr<FFT_IF> instance(); } }

using OTFFT_NAMESPACE::FFT_IF;
using Clock = std::chrono::steady_clock;
typedef std::unique_ptr<FFT_IF> (*AlgoFactory)();

// median over batches of one fwd+inv pair, in microseconds
static double timeIt(AlgoFactory f, int n, OTFFT::complex_t* x, OTFFT::complex_t* y)
{
    auto obj = f();
    int lg = 0; for (int m = 1 << n; m > 1; m >>= 1) lg++;
    obj->setup2(lg);
    const size_t N = size_t(1) << n;
    for (size_t i = 0; i < N; i++) { x[i].Re = std::cos(0.001*i); x[i].Im = std::sin(0.002*i); }
    // calibrate loops for ~20 ms batches
    int loops = 1;
    for (;;) {
        auto t1 = Clock::now();
        for (int j = 0; j < loops; j++) { obj->fwd(x, y); obj->inv(x, y); }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        if (ms >= 20.0 || loops >= (1 << 20)) break;
        loops *= 2;
    }
    std::vector<double> v;
    for (int r = 0; r < 9; r++) {
        auto t1 = Clock::now();
        for (int j = 0; j < loops; j++) { obj->fwd(x, y); obj->inv(x, y); }
        v.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t1).count() / loops);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

DECL(0)

int main()
{
    using namespace OTFFT_NAMESPACE;
    AlgoFactory fs[7] = { OTFFT_AVXDIF4::instance, OTFFT_AVXDIT4::instance, OTFFT_AVXDIF8::instance,
                      OTFFT_AVXDIT8::instance, OTFFT_AVXDIF16::instance, OTFFT_AVXDIT16::instance,
                      OTFFT_SixStep::instance };
    std::printf("n,DIF4,DIT4,DIF8,DIT8,DIF16,DIT16,SixStep,fastest\n");
    for (int n = 4; n <= 24; n++) {
        const size_t N = size_t(1) << n;
        auto* x = (OTFFT::complex_t*)simd_malloc(N * sizeof(OTFFT::complex_t));
        auto* y = (OTFFT::complex_t*)simd_malloc(N * sizeof(OTFFT::complex_t));
        double t[7]; int best = 0;
        for (int a = 0; a < 7; a++) { t[a] = timeIt(fs[a], n, x, y); if (t[a] < t[best]) best = a; }
        std::printf("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n", n, t[0], t[1], t[2], t[3], t[4], t[5], t[6], best + 1);
        std::fflush(stdout);
        simd_free(y); simd_free(x);
    }
}
