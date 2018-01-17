// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#include "otfft_config.h"
#include "otfft_platform.h"

#include <thread>
#include <array>
#include <stdexcept>
#include <cstdint>
#include <cassert>

#include <stdlib.h>
#include <string.h>

#if defined(PLAT_MINGW)
#include <cpuid.h>
#include <windows.h>
#elif defined(PLAT_WINDOWS)
#include <windows.h>
#include <limits.h>
#include <intrin.h>
#include <immintrin.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/param.h>
#include <sys/time.h>
#if defined(PLAT_LINUX)
#include <x86intrin.h>
#elif defined(PLAT_MACOS)
#include <sys/sysctl.h>
#endif
#endif

#include "otfft.h"
#include "otfft_misc.h"

#ifdef OTFFT_WITH_AVX2
#include "otfft_avx2.h"
#endif

#ifdef OTFFT_WITH_AVX
#include "otfft_avx.h"
#endif

#ifdef OTFFT_WITH_SSE2
#include "otfft_sse2.h"
#endif



namespace
{
    std::array<std::int32_t, 4> CPUID(std::int32_t info_type)
    {
        std::array<std::int32_t, 4> cpu_info;
#if defined(PLAT_LINUX) || defined(PLAT_MACOS)
#if defined(__pic__) && defined(__i386__)
        __asm__ volatile (
                    "mov %%ebx, %%edi\n"
                    "cpuid\n"
                    "xchg %%edi, %%ebx\n"
                    : "=a"(cpu_info[0]), "=D"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3]) : "a"(info_type), "c"(0));
#else
        __asm__ volatile (
                    "cpuid \n\t"
                    : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3]) : "a"(info_type), "c"(0));
#endif
#elif defined(PLAT_MINGW)
        __cpuid(info_type, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#elif defined(PLAT_WINDOWS)
        __cpuid(cpu_info.data(), info_type);
#endif
        return cpu_info;
    }

    std::uint64_t XGETBV(std::uint32_t xcr)
    {
#if defined(PLAT_WINDOWS) && !defined(PLAT_MINGW)
        return _xgetbv(xcr);
#else
        std::uint32_t eax, edx;
        __asm__ volatile ("xgetbv" : "=a" (eax), "=d" (edx) : "c" (xcr));
        return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
    }

    /**
     * @brief Compute system performance indicators and gather compile-time feature selection
     */
    struct sysinfo
    {
    private:
        /**
         * @brief Computes performance and gather feature selection (compiler)
         */
        sysinfo()
            : CPU_CORES(readNumberOfCores())
            , CPU_FEATURE_XSAVE(false)
            , CPU_FEATURE_SSE(false)
            , CPU_FEATURE_SSE2(false)
            , CPU_FEATURE_SSE3(false)
            , CPU_FEATURE_SSSE3(false)
            , CPU_FEATURE_SSE41(false)
            , CPU_FEATURE_SSE42(false)
            , CPU_FEATURE_AVX(false)
            , CPU_FEATURE_AVX2(false)
            , CPU_FEATURE_FMA(false)
        {
            const auto cpu_info_0 = CPUID(0);
            if (cpu_info_0[0] > 0)
            {
                const auto cpu_info_1 = CPUID(1);
                const auto cpu_info_xcr = XGETBV(0);

                CPU_FEATURE_XSAVE = ((cpu_info_1[2] & 0x04000000) != 0) && ((cpu_info_xcr & 6) == 6);
                CPU_FEATURE_SSE = (cpu_info_1[3] & 0x02000000) != 0;
                CPU_FEATURE_SSE2 = (cpu_info_1[3] & 0x04000000) != 0;
                CPU_FEATURE_SSE3 = (cpu_info_1[2] & 0x00000001) != 0;
                CPU_FEATURE_SSSE3 = (cpu_info_1[2] & 0x00000200) != 0;
                CPU_FEATURE_SSE41 = (cpu_info_1[2] & 0x00080000) != 0;
                CPU_FEATURE_SSE42 = (cpu_info_1[2] & 0x00100000) != 0;
                CPU_FEATURE_AVX = ((cpu_info_1[2] & 0x10000000) != 0) && ((cpu_info_1[2] & 0x04000000) != 0) && ((cpu_info_1[2] & 0x08000000) != 0) && CPU_FEATURE_XSAVE;
                CPU_FEATURE_FMA = (cpu_info_1[2] & 0x00001000) != 0;

                if (cpu_info_0[0] >= 7 && CPU_FEATURE_XSAVE)
                {
                    const auto cpu_info_7 = CPUID(7);

                    CPU_FEATURE_AVX2 = ((cpu_info_7[1] & 0x00000020) != 0);
                }
            }
        }

        /**
         * @brief Retrieves the number of logical CPU cores of the current system
         * @return the number of cores detected
         */
        std::size_t readNumberOfCores()
        {
            std::size_t num_cores = std::thread::hardware_concurrency();

            if (num_cores < 1)
            {
#if defined(PLAT_MINGW)
                // not possible - assume 2 to be safe
                num_cores = 2;
#elif defined(PLAT_WINDOWS)
                SYSTEM_INFO sysinfo;
                GetSystemInfo(&sysinfo);
                num_cores = sysinfo.dwNumberOfProcessors;
#elif defined(PLAT_MACOS)
                int nm[2];
                std::size_t len = 4;
                nm[0] = CTL_HW;
                nm[1] = HW_AVAILCPU;
                sysctl(nm, 2, &num_cores, &len, NULL, 0);

                if (num_cores < 1)
                {
                    nm[1] = HW_NCPU;
                    sysctl(nm, 2, &num_cores, &len, NULL, 0);

                    if(num_cores < 1)
                    {
                        num_cores = 1;
                    }
                }

#elif defined(PLAT_LINUX)
                num_cores = sysconf(_SC_NPROCESSORS_ONLN);
#endif
            }

            return num_cores;
        }

    public:
        /**
         * @brief Return the singleton instance
         */
        static inline const sysinfo* instance()
        {
            static const sysinfo instance;
            return &instance;
        }

        std::size_t CPU_CORES;

        bool CPU_FEATURE_XSAVE;
        bool CPU_FEATURE_SSE;
        bool CPU_FEATURE_SSE2;
        bool CPU_FEATURE_SSE3;
        bool CPU_FEATURE_SSSE3;
        bool CPU_FEATURE_SSE41;
        bool CPU_FEATURE_SSE42;
        bool CPU_FEATURE_AVX;
        bool CPU_FEATURE_AVX2;
        bool CPU_FEATURE_FMA;
    };
}

namespace OTFFT
{
    namespace CONSTANT
    {
        const double SQRT2 = 1.41421356237309504876378807303183294;
        const double PI = 3.14159265358979323846264338327950288;
        const double SQRT1_2 = 0.707106781186547524400844362104849039;
        const double RSQRT2PSQRT2 = 0.541196100146196984405268931572763336;
        const double H1X = 0.923879532511286762010323247995557949;
        const double H1Y = -0.382683432365089757574419179753100195;
    }

    void unique_ptr_delete::operator()(ComplexFFT *raw_pointer)
    {
#ifdef OTFFT_WITH_AVX2
            if (sysinfo::instance()->CPU_FEATURE_AVX2)
            {
                return OTFFT_AVX2::unique_ptr_deleter(raw_pointer);
            }
#endif

#ifdef OTFFT_WITH_AVX
            if (sysinfo::instance()->CPU_FEATURE_AVX)
            {
                return OTFFT_AVX::unique_ptr_deleter(raw_pointer);
            }
#endif

#ifdef OTFFT_WITH_SSE2
            if (sysinfo::instance()->CPU_FEATURE_SSE2)
            {
                return OTFFT_SSE2::unique_ptr_deleter(raw_pointer);
            }
#endif
            throw std::domain_error("No supported CPU featureset found.");
    }

    void unique_ptr_delete::operator()(RealFFT *raw_pointer)
    {
#ifdef OTFFT_WITH_AVX2
            if (sysinfo::instance()->CPU_FEATURE_AVX2)
            {
                return OTFFT_AVX2::unique_ptr_deleter(raw_pointer);
            }
#endif

#ifdef OTFFT_WITH_AVX
            if (sysinfo::instance()->CPU_FEATURE_AVX)
            {
                return OTFFT_AVX::unique_ptr_deleter(raw_pointer);
            }
#endif

#ifdef OTFFT_WITH_SSE2
            if (sysinfo::instance()->CPU_FEATURE_SSE2)
            {
                return OTFFT_SSE2::unique_ptr_deleter(raw_pointer);
            }
#endif
            throw std::domain_error("No supported CPU featureset found.");
    }

    void unique_ptr_delete::operator()(RealDCT *raw_pointer)
    {
#ifdef OTFFT_WITH_AVX2
            if (sysinfo::instance()->CPU_FEATURE_AVX2)
            {
                return OTFFT_AVX2::unique_ptr_deleter(raw_pointer);
            }
#endif

#ifdef OTFFT_WITH_AVX
            if (sysinfo::instance()->CPU_FEATURE_AVX)
            {
                return OTFFT_AVX::unique_ptr_deleter(raw_pointer);
            }
#endif

#ifdef OTFFT_WITH_SSE2
            if (sysinfo::instance()->CPU_FEATURE_SSE2)
            {
                return OTFFT_SSE2::unique_ptr_deleter(raw_pointer);
            }
#endif
            throw std::domain_error("No supported CPU featureset found.");
    }

    namespace Factory
    {
        ComplexFFTPtr createComplexFFT(int n, OptimizationType t)
        {
#ifdef OTFFT_WITH_AVX2
            if (sysinfo::instance()->CPU_FEATURE_AVX2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX2))
            {
                return ComplexFFTPtr(OTFFT_AVX2::Factory::createComplexFFT(n));
            }
#endif

#ifdef OTFFT_WITH_AVX
            if (sysinfo::instance()->CPU_FEATURE_AVX &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX))
            {
                return ComplexFFTPtr(OTFFT_AVX::Factory::createComplexFFT(n));
            }
#endif

#ifdef OTFFT_WITH_SSE2
            if (sysinfo::instance()->CPU_FEATURE_SSE2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_SSE2))
            {
                return ComplexFFTPtr(OTFFT_SSE2::Factory::createComplexFFT(n));
            }
#endif
            throw std::domain_error("No supported CPU featureset found.");
        }

        RealFFTPtr createRealFFT(int n, OptimizationType t)
        {
#ifdef OTFFT_WITH_AVX2
            if (sysinfo::instance()->CPU_FEATURE_AVX2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX2))
            {
                return RealFFTPtr(OTFFT_AVX2::Factory::createRealFFT(n));
            }
#endif

#ifdef OTFFT_WITH_AVX
            if (sysinfo::instance()->CPU_FEATURE_AVX &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX))
            {
                return RealFFTPtr(OTFFT_AVX::Factory::createRealFFT(n));
            }
#endif

#ifdef OTFFT_WITH_SSE2
            if (sysinfo::instance()->CPU_FEATURE_SSE2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_SSE2))
            {
                return RealFFTPtr(OTFFT_SSE2::Factory::createRealFFT(n));
            }
#endif
            throw std::domain_error("No supported CPU featureset found.");
        }

        RealDCTPtr createDCT(int n, OptimizationType t)
        {
#ifdef OTFFT_WITH_AVX2
            if (sysinfo::instance()->CPU_FEATURE_AVX2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX2))
            {
                return RealDCTPtr(OTFFT_AVX2::Factory::createDCT(n));
            }
#endif

#ifdef OTFFT_WITH_AVX
            if (sysinfo::instance()->CPU_FEATURE_AVX &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX))
            {
                return RealDCTPtr(OTFFT_AVX::Factory::createDCT(n));
            }
#endif

#ifdef OTFFT_WITH_SSE2
            if (sysinfo::instance()->CPU_FEATURE_SSE2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_SSE2))
            {
                return RealDCTPtr(OTFFT_SSE2::Factory::createDCT(n));
            }
#endif

            throw std::domain_error("No supported CPU featureset found.");
        }

        ComplexFFTPtr createBluesteinFFT(int n, OptimizationType t)
        {
#ifdef OTFFT_WITH_AVX2
            if (sysinfo::instance()->CPU_FEATURE_AVX2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX2))
            {
                return ComplexFFTPtr(OTFFT_AVX2::Factory::createBluesteinFFT(n));
            }
#endif

#ifdef OTFFT_WITH_AVX
            if (sysinfo::instance()->CPU_FEATURE_AVX &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_AVX))
            {
                return ComplexFFTPtr(OTFFT_AVX::Factory::createBluesteinFFT(n));
            }
#endif

#ifdef OTFFT_WITH_SSE2
            if (sysinfo::instance()->CPU_FEATURE_SSE2 &&
                (t == OptimizationType::OPTIMIZED_FFT_AUTO ||
                 t == OptimizationType::OPTIMIZED_FFT_SSE2))
            {
                return ComplexFFTPtr(OTFFT_SSE2::Factory::createBluesteinFFT(n));
            }
#endif
            throw std::domain_error("No supported CPU featureset found.");
        }
    }
}
