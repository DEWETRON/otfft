// Part of the ffttune tool. This file is compiled once per CPU backend (SSE2, AVX, AVX2)
// with the same flags as the library's algorithm sources; FFTTUNE_MEASURE names the
// function it exports (for example ffttune_measure_avx2).
#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"

#ifdef USE_AVX2
#include "otfft_avx2.h"
#endif
#ifdef USE_AVX
#include "otfft_avx.h"
#endif
#ifdef USE_SSE2
#include "otfft_sse2.h"
#endif

#include "ffttune.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace OTFFT_NAMESPACE
{
    namespace OTFFT_AVXDIF4 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT4 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF8 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT8 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_SixStep { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_TwoPhase { extern std::unique_ptr<FFT_IF> instance(); }
}

namespace
{
    typedef std::chrono::steady_clock Clock;
    typedef std::unique_ptr<OTFFT_NAMESPACE::FFT_IF> (*AlgoFactory)();

    void fill(OTFFT::complex_t* x, std::size_t n)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            const double t = double(i) / double(n);
            x[i].Re = 10 * std::cos(3 * 2 * 3.14159265358979323846 * t * t);
            x[i].Im = 10 * std::sin(3 * 2 * 3.14159265358979323846 * t * t);
        }
    }

    double elapsedUs(Clock::time_point t1)
    {
        return std::chrono::duration<double, std::micro>(Clock::now() - t1).count();
    }
}

bool FFTTUNE_MEASURE(int log_n, int repeats, double batch_ms, double us[ffttune::kAlgorithms])
{
    using namespace OTFFT_NAMESPACE;
    const AlgoFactory factories[ffttune::kAlgorithms] = {
        OTFFT_AVXDIF4::instance, OTFFT_AVXDIT4::instance, OTFFT_AVXDIF8::instance, OTFFT_AVXDIT8::instance,
        OTFFT_AVXDIF16::instance, OTFFT_AVXDIT16::instance, OTFFT_SixStep::instance, OTFFT_TwoPhase::instance};

    const std::size_t n = std::size_t(1) << log_n;
    auto* x = static_cast<OTFFT::complex_t*>(simd_malloc(n * sizeof(OTFFT::complex_t)));
    auto* y = static_cast<OTFFT::complex_t*>(simd_malloc(n * sizeof(OTFFT::complex_t)));
    if (!x || !y)
    {
        if (y) simd_free(y);
        if (x) simd_free(x);
        return false;
    }

    std::vector<OTFFT::complex_t> reference(n);
    double referenceNorm = 0;
    for (int a = 0; a < ffttune::kAlgorithms; a++)
    {
        us[a] = -1;
        try
        {
            auto obj = factories[a]();
            obj->setup2(log_n);

            // result check against DIF4 before anything is timed
            fill(x, n);
            obj->fwd(x, y);
            if (a == 0)
            {
                std::copy(x, x + n, reference.begin());
                for (std::size_t i = 0; i < n; i++)
                {
                    referenceNorm += x[i].Re * x[i].Re + x[i].Im * x[i].Im;
                }
            }
            else
            {
                double diff = 0;
                for (std::size_t i = 0; i < n; i++)
                {
                    const double dr = x[i].Re - reference[i].Re;
                    const double di = x[i].Im - reference[i].Im;
                    diff += dr * dr + di * di;
                }
                if (!(std::sqrt(diff / referenceNorm) <= 1e-9))
                {
                    continue;
                }
            }

            // loops per batch, so that one batch takes about batch_ms
            int loops = 1;
            for (;;)
            {
                fill(x, n);
                const Clock::time_point t1 = Clock::now();
                for (int j = 0; j < loops; j++)
                {
                    obj->fwd(x, y);
                    obj->inv(x, y);
                }
                if (elapsedUs(t1) >= batch_ms * 1000.0 || loops >= (1 << 20))
                {
                    break;
                }
                loops *= 2;
            }

            std::vector<double> samples;
            for (int r = 0; r < repeats; r++)
            {
                fill(x, n);
                const Clock::time_point t1 = Clock::now();
                for (int j = 0; j < loops; j++)
                {
                    obj->fwd(x, y);
                    obj->inv(x, y);
                }
                samples.push_back(elapsedUs(t1) / loops);
            }
            std::sort(samples.begin(), samples.end());
            us[a] = samples[samples.size() / 2];
        }
        catch (...)
        {
            us[a] = -1;
        }
    }
    simd_free(y);
    simd_free(x);
    return true;
}
