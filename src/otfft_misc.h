/******************************************************************************
*  FFT Miscellaneous Routines Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/
// Copyright (c) 2017 to the present, DEWETRON GmbH

#ifndef otfft_misc_h
#define otfft_misc_h

//=============================================================================
// Customization Options
//=============================================================================

#define USE_INTRINSIC 1
//#define DO_SINGLE_THREAD 1
//#define USE_UNALIGNED_MEMORY 1

//=============================================================================

#include "otfft_types.h"
#include <complex>
#include <cmath>
#include <new>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504876378807303183294
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039
#endif

namespace OTFFT_MISC {
    using namespace OTFFT;

//using namespace OTFFT_Complex;

constexpr double H1X =  0.923879532511286762010323247995557949;
constexpr double H1Y = -0.382683432365089757574419179753100195;

enum scaling_mode { scale_1 = 0, scale_unitary, scale_length };
} // namespace OTFFT_MISC

#ifdef _MSC_VER
//=============================================================================
// for Visual C++
//=============================================================================

#if _M_IX86_FP >= 2
#define __SSE2__ 1
#endif
#ifdef _M_X64
#define __SSE2__ 1
#define __SSE3__ 1
#endif
#ifdef __AVX__
#define __SSE2__ 1
#define __SSE3__ 1
#endif
#ifdef __AVX2__
#define __SSE2__ 1
#define __SSE3__ 1
#define __FMA__ 1
#endif
#ifdef __AVX512F__
#define __SSE2__ 1
#define __SSE3__ 1
#define __FMA__ 1
#endif

#if _MSC_VER >= 1900
#define VC_CONSTEXPR 1
#else
#error "This compiler is not supported."
#endif

#endif // _MSC_VER

//=============================================================================
// FFT Weight Initialize Routine
//=============================================================================

namespace OTFFT_MISC {

#ifdef DO_SINGLE_THREAD
constexpr int OMP_THRESHOLD_W = 1<<30;
#else
//constexpr int OMP_THRESHOLD_W = 1<<16;
constexpr int OMP_THRESHOLD_W = 1<<15;
#endif

    static void init_W(const int N, complex_vector W) noexcept
    {
        const double theta0 = 2*M_PI/N;
        const int Nh = N/2;
        const int Nq = N/4;
        const int Ne = N/8;
        const int Nd = N - Nq;
        if (N < 1) {}
        else if (N < 2) { W[0] = W[1] = 1; }
        else if (N < 4) { W[0] = W[2] = 1; W[1] = -1; }
        else if (N < 8) {
            W[0] = complex_t( 1,  0);
            W[1] = complex_t( 0, -1);
            W[2] = complex_t(-1,  0);
            W[3] = complex_t( 0,  1);
            W[4] = complex_t( 1,  0);
        }
        else if (N < OMP_THRESHOLD_W) for (int p = 0; p <= Ne; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            W[p]    = complex_t( c,  s);
            W[Nq-p] = complex_t(-s, -c);
            W[Nq+p] = complex_t( s, -c);
            W[Nh-p] = complex_t(-c,  s);
            W[Nh+p] = complex_t(-c, -s);
            W[Nd-p] = complex_t( s,  c);
            W[Nd+p] = complex_t(-s,  c);
            W[N-p]  = complex_t( c, -s);
        }
        else {
        #pragma omp parallel for schedule(static)
        for (int p = 0; p <= Ne; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            W[p]    = complex_t( c,  s);
            W[Nq-p] = complex_t(-s, -c);
            W[Nq+p] = complex_t( s, -c);
            W[Nh-p] = complex_t(-c,  s);
            W[Nh+p] = complex_t(-c, -s);
            W[Nd-p] = complex_t( s,  c);
            W[Nd+p] = complex_t(-s,  c);
            W[N-p]  = complex_t( c, -s);
        }
        }
    }

    static void speedup_magic(const int N = 1 << 18) noexcept
    {
        const double theta0 = 2*M_PI/N;
        volatile double sum = 0;
        for (int p = 0; p < N; p++) {
            sum += cos(p * theta0);
        }
    }

} // namespace OTFFT_MISC

#if defined(__SSE2__) && defined(USE_INTRINSIC)
//=============================================================================
// SSE2/SSE3
//=============================================================================

#include <emmintrin.h>

namespace OTFFT_MISC {

    typedef __m128d xmm;

    static inline xmm cmplx(const double& x, const double& y) noexcept force_inline;
    static inline xmm cmplx(const double& x, const double& y) noexcept
    {
        return _mm_setr_pd(x, y);
    }

    static inline xmm getpz(const complex_t& z) noexcept force_inline;
    static inline xmm getpz(const complex_t& z) noexcept
    {
#ifdef USE_UNALIGNED_MEMORY
        return _mm_loadu_pd(&z.Re);
#else
        return _mm_load_pd(&z.Re);
#endif
    }
    static inline xmm getpz(const_double_vector x) noexcept force_inline;
    static inline xmm getpz(const_double_vector x) noexcept
    {
#ifdef USE_UNALIGNED_MEMORY
        return _mm_loadu_pd(x);
#else
        return _mm_load_pd(x);
#endif
    }

    static inline void setpz(complex_t& z, const xmm x) noexcept force_inline3;
    static inline void setpz(complex_t& z, const xmm x) noexcept
    {
#ifdef USE_UNALIGNED_MEMORY
        _mm_storeu_pd(&z.Re, x);
#else
        _mm_store_pd(&z.Re, x);
#endif
    }
    static inline void setpz(double_vector x, const xmm z) noexcept force_inline3;
    static inline void setpz(double_vector x, const xmm z) noexcept
    {
#ifdef USE_UNALIGNED_MEMORY
        _mm_storeu_pd(x, z);
#else
        _mm_store_pd(x, z);
#endif
    }

    static inline void swappz(complex_t& x, complex_t& y) noexcept force_inline3;
    static inline void swappz(complex_t& x, complex_t& y) noexcept
    {
        const xmm z = getpz(x); setpz(x, getpz(y)); setpz(y, z);
    }

    static inline xmm cnjpz(const xmm xy) noexcept force_inline;
    static inline xmm cnjpz(const xmm xy) noexcept
    {
        constexpr xmm zm = { 0.0, -0.0 };
        return _mm_xor_pd(zm, xy);
    }
    static inline xmm jxpz(const xmm xy) noexcept force_inline;
    static inline xmm jxpz(const xmm xy) noexcept
    {
        const xmm xmy = cnjpz(xy);
        return _mm_shuffle_pd(xmy, xmy, 1);
    }
    static inline xmm negpz(const xmm xy) noexcept force_inline;
    static inline xmm negpz(const xmm xy) noexcept
    {
        constexpr xmm mm = { -0.0, -0.0 };
        return _mm_xor_pd(mm, xy);
    }
    static inline xmm mjxpz(const xmm xy) noexcept force_inline;
    static inline xmm mjxpz(const xmm xy) noexcept
    {
        const xmm yx = _mm_shuffle_pd(xy, xy, 1);
        return cnjpz(yx);
    }

    static inline xmm addpz(const xmm a, const xmm b) noexcept force_inline;
    static inline xmm addpz(const xmm a, const xmm b) noexcept
    {
        return _mm_add_pd(a, b);
    }
    static inline xmm subpz(const xmm a, const xmm b) noexcept force_inline;
    static inline xmm subpz(const xmm a, const xmm b) noexcept
    {
        return _mm_sub_pd(a, b);
    }
    static inline xmm mulpd(const xmm a, const xmm b) noexcept force_inline;
    static inline xmm mulpd(const xmm a, const xmm b) noexcept
    {
        return _mm_mul_pd(a, b);
    }
    static inline xmm divpd(const xmm a, const xmm b) noexcept force_inline;
    static inline xmm divpd(const xmm a, const xmm b) noexcept
    {
        return _mm_div_pd(a, b);
    }

    template <int N, int mode> static inline xmm scalepz(const xmm z) noexcept force_inline;
    template <int N, int mode> static inline xmm scalepz(const xmm z) noexcept
    {
        constexpr double scale =
            mode == scale_1       ? 1.0           :
            mode == scale_unitary ? 1.0/mysqrt(N) :
            mode == scale_length  ? 1.0/N         : 0.0;
        constexpr xmm sv = { scale, scale };
        return mode == scale_1 ? z : mulpd(sv, z);
    }

} // namespace OTFFT_MISC

#if defined(__SSE3__)
//-----------------------------------------------------------------------------
// SSE3
//-----------------------------------------------------------------------------

#ifdef __FMA__
#include <immintrin.h>
#else
#include <pmmintrin.h>
#endif

namespace OTFFT_MISC {

    static inline xmm haddpz(const xmm ab, const xmm xy) noexcept force_inline;
    static inline xmm haddpz(const xmm ab, const xmm xy) noexcept
    {
        return _mm_hadd_pd(ab, xy); // (a + b, x + y)
    }

    static inline xmm mulpz(const xmm ab, const xmm xy) noexcept force_inline;
    static inline xmm mulpz(const xmm ab, const xmm xy) noexcept
    {
        const xmm aa = _mm_unpacklo_pd(ab, ab);
        const xmm bb = _mm_unpackhi_pd(ab, ab);
        const xmm yx = _mm_shuffle_pd(xy, xy, 1);
#ifdef __FMA__
        return _mm_fmaddsub_pd(aa, xy, _mm_mul_pd(bb, yx));
#else
        return _mm_addsub_pd(_mm_mul_pd(aa, xy), _mm_mul_pd(bb, yx));
#endif
    }

    static inline xmm divpz(const xmm ab, const xmm xy) noexcept force_inline;
    static inline xmm divpz(const xmm ab, const xmm xy) noexcept
    {
        const xmm x2y2 = _mm_mul_pd(xy, xy);
        const xmm r2r2 = _mm_hadd_pd(x2y2, x2y2);
        return _mm_div_pd(mulpz(ab, cnjpz(xy)), r2r2);
    }

    static inline xmm v8xpz(const xmm xy) noexcept force_inline;
    static inline xmm v8xpz(const xmm xy) noexcept
    {
        constexpr xmm rr = { M_SQRT1_2, M_SQRT1_2 };
        const xmm yx = _mm_shuffle_pd(xy, xy, 1);
        return _mm_mul_pd(rr, _mm_addsub_pd(xy, yx));
    }

} // namespace OTFFT_MISC

#else
//-----------------------------------------------------------------------------
// SSE3 Emulation
//-----------------------------------------------------------------------------

namespace OTFFT_MISC {

    static inline xmm haddpz(const xmm ab, const xmm xy) noexcept force_inline;
    static inline xmm haddpz(const xmm ab, const xmm xy) noexcept
    {
        const xmm ba = _mm_shuffle_pd(ab, ab, 1);
        const xmm yx = _mm_shuffle_pd(xy, xy, 1);
        const xmm apb = _mm_add_sd(ab, ba);
        const xmm xpy = _mm_add_sd(xy, yx);
        return _mm_shuffle_pd(apb, xpy, 0); // (a + b, x + y)
    }

    static inline xmm mulpz(const xmm ab, const xmm xy) noexcept force_inline;
    static inline xmm mulpz(const xmm ab, const xmm xy) noexcept
    {
        const xmm aa = _mm_unpacklo_pd(ab, ab);
        const xmm bb = _mm_unpackhi_pd(ab, ab);
        return _mm_add_pd(_mm_mul_pd(aa, xy), _mm_mul_pd(bb, jxpz(xy)));
    }

    static inline xmm divpz(const xmm ab, const xmm xy) noexcept force_inline;
    static inline xmm divpz(const xmm ab, const xmm xy) noexcept
    {
        const xmm x2y2 = _mm_mul_pd(xy, xy);
        const xmm y2x2 = _mm_shuffle_pd(x2y2, x2y2, 1);
        const xmm r2r2 = _mm_add_pd(x2y2, y2x2);
        return _mm_div_pd(mulpz(ab, cnjpz(xy)), r2r2);
    }

    static inline xmm v8xpz(const xmm xy) noexcept force_inline;
    static inline xmm v8xpz(const xmm xy) noexcept
    {
        constexpr xmm rr = { M_SQRT1_2, M_SQRT1_2 };
        return _mm_mul_pd(rr, _mm_add_pd(xy, jxpz(xy)));
    }

} // namespace OTFFT_MISC

//-----------------------------------------------------------------------------
#endif // __SSE3__

namespace OTFFT_MISC {

    static inline xmm w8xpz(const xmm xy) noexcept force_inline;
    static inline xmm w8xpz(const xmm xy) noexcept
    {
        constexpr xmm rr = { M_SQRT1_2, M_SQRT1_2 };
        const xmm ymx = cnjpz(_mm_shuffle_pd(xy, xy, 1));
        return _mm_mul_pd(rr, _mm_add_pd(xy, ymx));
    }

    static inline xmm h1xpz(const xmm xy) noexcept force_inline;
    static inline xmm h1xpz(const xmm xy) noexcept
    {
        constexpr xmm h1 = { H1X, H1Y };
        return mulpz(h1, xy);
    }

    static inline xmm h3xpz(const xmm xy) noexcept force_inline;
    static inline xmm h3xpz(const xmm xy) noexcept
    {
        constexpr xmm h3 = { -H1Y, -H1X };
        return mulpz(h3, xy);
    }

    static inline xmm hfxpz(const xmm xy) noexcept force_inline;
    static inline xmm hfxpz(const xmm xy) noexcept
    {
        constexpr xmm hf = { H1X, -H1Y };
        return mulpz(hf, xy);
    }

    static inline xmm hdxpz(const xmm xy) noexcept force_inline;
    static inline xmm hdxpz(const xmm xy) noexcept
    {
        constexpr xmm hd = { -H1Y, H1X };
        return mulpz(hd, xy);
    }

#if !defined(USE_AVX) && !defined(USE_AVX2)
    static inline void* simd_malloc(const size_t n) { return _mm_malloc(n, 16); }
    static inline void simd_free(void* p) { _mm_free(p); }
#endif

} // namespace OTFFT_MISC

#else
//=============================================================================
// SSE2/SSE3 Emulation
//=============================================================================

namespace OTFFT_MISC {

    struct xmm { double Re, Im; };

    static inline xmm cmplx(const double& x, const double& y) noexcept force_inline;
    static inline xmm cmplx(const double& x, const double& y) noexcept
    {
        const xmm z = { x, y };
        return z;
    }

    static inline xmm getpz(const complex_t& z) noexcept force_inline;
    static inline xmm getpz(const complex_t& z) noexcept
    {
        const xmm x = { z.Re, z.Im };
        return x;
    }
    static inline xmm getpz(const_double_vector x) noexcept force_inline;
    static inline xmm getpz(const_double_vector x) noexcept
    {
        const xmm z = { x[0], x[1] };
        return z;
    }

    static inline void setpz(complex_t& z, const xmm& x) noexcept force_inline3;
    static inline void setpz(complex_t& z, const xmm& x) noexcept
    {
        z.Re = x.Re; z.Im = x.Im;
    }
    static inline void setpz(double_vector x, const xmm z) noexcept force_inline3;
    static inline void setpz(double_vector x, const xmm z) noexcept
    {
        x[0] = z.Re; x[1] = z.Im;
    }

    static inline void swappz(complex_t& x, complex_t& y) noexcept force_inline3;
    static inline void swappz(complex_t& x, complex_t& y) noexcept
    {
        const xmm z = getpz(x); setpz(x, getpz(y)); setpz(y, z);
    }

    static inline xmm cnjpz(const xmm& z) noexcept force_inline;
    static inline xmm cnjpz(const xmm& z) noexcept
    {
        const xmm x = { z.Re, -z.Im };
        return x;
    }
    static inline xmm jxpz(const xmm& z) noexcept force_inline;
    static inline xmm jxpz(const xmm& z) noexcept
    {
        const xmm x = { -z.Im, z.Re };
        return x;
    }
    static inline xmm negpz(const xmm& z) noexcept force_inline;
    static inline xmm negpz(const xmm& z) noexcept
    {
        const xmm x = { -z.Re, -z.Im };
        return x;
    }
    static inline xmm mjxpz(const xmm& z) noexcept force_inline;
    static inline xmm mjxpz(const xmm& z) noexcept
    {
        const xmm x = { z.Im, -z.Re };
        return x;
    }

    static inline xmm addpz(const xmm& a, const xmm& b) noexcept force_inline;
    static inline xmm addpz(const xmm& a, const xmm& b) noexcept
    {
        const xmm x = { a.Re + b.Re, a.Im + b.Im };
        return x;
    }
    static inline xmm subpz(const xmm& a, const xmm& b) noexcept force_inline;
    static inline xmm subpz(const xmm& a, const xmm& b) noexcept
    {
        const xmm x = { a.Re - b.Re, a.Im - b.Im };
        return x;
    }
    static inline xmm mulpd(const xmm& a, const xmm& b) noexcept force_inline;
    static inline xmm mulpd(const xmm& a, const xmm& b) noexcept
    {
        const xmm x = { a.Re*b.Re, a.Im*b.Im };
        return x;
    }
    static inline xmm divpd(const xmm& a, const xmm& b) noexcept force_inline;
    static inline xmm divpd(const xmm& a, const xmm& b) noexcept
    {
        const xmm x = { a.Re/b.Re, a.Im/b.Im };
        return x;
    }

    template <int N, int mode> static inline xmm scalepz(const xmm z) noexcept force_inline;
    template <int N, int mode> static inline xmm scalepz(const xmm z) noexcept
    {
        constexpr double scale =
            mode == scale_1       ? 1.0           :
            mode == scale_unitary ? 1.0/mysqrt(N) :
            mode == scale_length  ? 1.0/N         : 0.0;
        constexpr xmm sv = { scale, scale };
        return mode == scale_1 ? z : mulpd(sv, z);
    }

    static inline xmm mulpz(const xmm& a, const xmm& b) noexcept force_inline;
    static inline xmm mulpz(const xmm& a, const xmm& b) noexcept
    {
        const xmm x = { a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re };
        return x;
    }

    static inline xmm divpz(const xmm& a, const xmm& b) noexcept force_inline;
    static inline xmm divpz(const xmm& a, const xmm& b) noexcept
    {
        const double b2 = b.Re*b.Re + b.Im*b.Im;
        const xmm acb = mulpz(a, cnjpz(b));
        const xmm x = { acb.Re/b2, acb.Im/b2 };
        return x;
    }

    static inline xmm haddpz(const xmm& ab, const xmm& xy) noexcept force_inline;
    static inline xmm haddpz(const xmm& ab, const xmm& xy) noexcept
    {
        const xmm x = { ab.Re + ab.Im, xy.Re + xy.Im };
        return x;
    }

    static inline xmm v8xpz(const xmm& z) noexcept force_inline;
    static inline xmm v8xpz(const xmm& z) noexcept
    {
        const xmm x = { M_SQRT1_2*(z.Re - z.Im), M_SQRT1_2*(z.Re + z.Im) };
        return x;
    }

    static inline xmm w8xpz(const xmm& z) noexcept force_inline;
    static inline xmm w8xpz(const xmm& z) noexcept
    {
        const xmm x = { M_SQRT1_2*(z.Re + z.Im), M_SQRT1_2*(z.Im - z.Re) };
        return x;
    }

    static inline xmm h1xpz(const xmm xy) noexcept force_inline;
    static inline xmm h1xpz(const xmm xy) noexcept
    {
        constexpr xmm h1 = { H1X, H1Y };
        return mulpz(h1, xy);
    }

    static inline xmm h3xpz(const xmm xy) noexcept force_inline;
    static inline xmm h3xpz(const xmm xy) noexcept
    {
        constexpr xmm h3 = { -H1Y, -H1X };
        return mulpz(h3, xy);
    }

    static inline xmm hfxpz(const xmm xy) noexcept force_inline;
    static inline xmm hfxpz(const xmm xy) noexcept
    {
        constexpr xmm hf = { H1X, -H1Y };
        return mulpz(hf, xy);
    }

    static inline xmm hdxpz(const xmm xy) noexcept force_inline;
    static inline xmm hdxpz(const xmm xy) noexcept
    {
        constexpr xmm hd = { -H1Y, H1X };
        return mulpz(hd, xy);
    }

    static inline void* simd_malloc(const size_t n) { return malloc(n); }
    static inline void simd_free(void* p) { free(p); }

} // namespace OTFFT_MISC

#endif // __SSE2__

#if defined(__AVX__) && defined(USE_INTRINSIC)
//=============================================================================
// AVX
//=============================================================================

#include <immintrin.h>

namespace OTFFT_MISC {

    typedef __m256d ymm;

    static inline void zeroupper() noexcept { _mm256_zeroupper(); }

    static inline ymm cmplx2(const double a, const double b, const double c, const double d) noexcept force_inline;
    static inline ymm cmplx2(const double a, const double b, const double c, const double d) noexcept
    {
        return _mm256_setr_pd(a, b, c, d);
    }

    static inline ymm cmplx2(const complex_t& x, const complex_t& y) noexcept force_inline;
    static inline ymm cmplx2(const complex_t& x, const complex_t& y) noexcept
    {
#if 0
        const xmm a = getpz(x);
        const xmm b = getpz(y);
        const ymm ax = _mm256_castpd128_pd256(a);
        const ymm bx = _mm256_castpd128_pd256(b);
        return _mm256_permute2f128_pd(ax, bx, 0x20);
#else
        return _mm256_setr_pd(x.Re, x.Im, y.Re, y.Im);
#endif
    }

    static inline ymm getpz2(const_complex_vector z) noexcept force_inline;
    static inline ymm getpz2(const_complex_vector z) noexcept
    {
#ifdef USE_UNALIGNED_MEMORY
        return _mm256_loadu_pd(&z->Re);
#else
        return _mm256_load_pd(&z->Re);
#endif
    }

    static inline void setpz2(complex_vector z, const ymm x) noexcept force_inline3;
    static inline void setpz2(complex_vector z, const ymm x) noexcept
    {
#ifdef USE_UNALIGNED_MEMORY
        _mm256_storeu_pd(&z->Re, x);
#else
        _mm256_store_pd(&z->Re, x);
#endif
    }

    static inline ymm cnjpz2(const ymm xy) noexcept force_inline;
    static inline ymm cnjpz2(const ymm xy) noexcept
    {
        constexpr ymm zm = { 0.0, -0.0, 0.0, -0.0 };
        return _mm256_xor_pd(zm, xy);
    }
    static inline ymm jxpz2(const ymm xy) noexcept force_inline;
    static inline ymm jxpz2(const ymm xy) noexcept
    {
        const ymm xmy = cnjpz2(xy);
        return _mm256_shuffle_pd(xmy, xmy, 5);
    }
    static inline ymm negpz2(const ymm xy) noexcept force_inline;
    static inline ymm negpz2(const ymm xy) noexcept
    {
        constexpr ymm mm = { -0.0, -0.0, -0.0, -0.0 };
        return _mm256_xor_pd(mm, xy);
    }

    static inline ymm addpz2(const ymm a, const ymm b) noexcept force_inline;
    static inline ymm addpz2(const ymm a, const ymm b) noexcept
    {
        return _mm256_add_pd(a, b);
    }
    static inline ymm subpz2(const ymm a, const ymm b) noexcept force_inline;
    static inline ymm subpz2(const ymm a, const ymm b) noexcept
    {
        return _mm256_sub_pd(a, b);
    }
    static inline ymm mulpd2(const ymm a, const ymm b) noexcept force_inline;
    static inline ymm mulpd2(const ymm a, const ymm b) noexcept
    {
        return _mm256_mul_pd(a, b);
    }
    static inline ymm divpd2(const ymm a, const ymm b) noexcept force_inline;
    static inline ymm divpd2(const ymm a, const ymm b) noexcept
    {
        return _mm256_div_pd(a, b);
    }

    static inline ymm mulpz2(const ymm ab, const ymm xy) noexcept force_inline;
    static inline ymm mulpz2(const ymm ab, const ymm xy) noexcept
    {
        const ymm aa = _mm256_unpacklo_pd(ab, ab);
        const ymm bb = _mm256_unpackhi_pd(ab, ab);
        const ymm yx = _mm256_shuffle_pd(xy, xy, 5);
#ifdef __FMA__
        return _mm256_fmaddsub_pd(aa, xy, _mm256_mul_pd(bb, yx));
#else
        return _mm256_addsub_pd(_mm256_mul_pd(aa, xy), _mm256_mul_pd(bb, yx));
#endif
    }

    static inline ymm divpz2(const ymm ab, const ymm xy) noexcept force_inline;
    static inline ymm divpz2(const ymm ab, const ymm xy) noexcept
    {
        const ymm x2y2 = _mm256_mul_pd(xy, xy);
        const ymm r2r2 = _mm256_hadd_pd(x2y2, x2y2);
        return _mm256_div_pd(mulpz2(ab, cnjpz2(xy)), r2r2);
    }

    template <int N, int mode> static inline ymm scalepz2(const ymm z) noexcept force_inline;
    template <int N, int mode> static inline ymm scalepz2(const ymm z) noexcept
    {
        constexpr double scale =
            mode == scale_1       ? 1.0           :
            mode == scale_unitary ? 1.0/mysqrt(N) :
            mode == scale_length  ? 1.0/N         : 0.0;
        constexpr ymm sv = { scale, scale, scale, scale };
        return mode == scale_1 ? z : mulpd2(sv, z);
    }

    static inline ymm v8xpz2(const ymm xy) noexcept force_inline;
    static inline ymm v8xpz2(const ymm xy) noexcept
    {
        constexpr ymm rr = { M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2 };
        const ymm yx = _mm256_shuffle_pd(xy, xy, 5);
        return _mm256_mul_pd(rr, _mm256_addsub_pd(xy, yx));
    }

    static inline ymm w8xpz2(const ymm xy) noexcept force_inline;
    static inline ymm w8xpz2(const ymm xy) noexcept
    {
        constexpr ymm rr = { M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2 };
        const ymm ymx = cnjpz2(_mm256_shuffle_pd(xy, xy, 5));
        return _mm256_mul_pd(rr, _mm256_add_pd(xy, ymx));
    }

    static inline ymm h1xpz2(const ymm xy) noexcept force_inline;
    static inline ymm h1xpz2(const ymm xy) noexcept
    {
        constexpr ymm h1 = { H1X, H1Y, H1X, H1Y };
        return mulpz2(h1, xy);
    }

    static inline ymm h3xpz2(const ymm xy) noexcept force_inline;
    static inline ymm h3xpz2(const ymm xy) noexcept
    {
        constexpr ymm h3 = { -H1Y, -H1X, -H1Y, -H1X };
        return mulpz2(h3, xy);
    }

    static inline ymm hfxpz2(const ymm xy) noexcept force_inline;
    static inline ymm hfxpz2(const ymm xy) noexcept
    {
        constexpr ymm hf = { H1X, -H1Y, H1X, -H1Y };
        return mulpz2(hf, xy);
    }

    static inline ymm hdxpz2(const ymm xy) noexcept force_inline;
    static inline ymm hdxpz2(const ymm xy) noexcept
    {
        constexpr ymm hd = { -H1Y, H1X, -H1Y, H1X };
        return mulpz2(hd, xy);
    }

    static inline ymm duppz2(const xmm x) noexcept force_inline;
    static inline ymm duppz2(const xmm x) noexcept
    {
        return _mm256_broadcast_pd(&x);
    }

    static inline ymm duppz3(const complex_t& z) noexcept force_inline;
    static inline ymm duppz3(const complex_t& z) noexcept
    {
        return _mm256_broadcast_pd(reinterpret_cast<const xmm *>(&z));
    }

    static inline ymm cat(const xmm a, const xmm b) noexcept force_inline;
    static inline ymm cat(const xmm a, const xmm b) noexcept
    {
        const ymm ax = _mm256_castpd128_pd256(a);
        //const ymm bx = _mm256_castpd128_pd256(b);
        //return _mm256_permute2f128_pd(ax, bx, 0x20);
        return _mm256_insertf128_pd(ax, b, 1);
    }

    static inline ymm catlo(const ymm ax, const ymm by) noexcept force_inline;
    static inline ymm catlo(const ymm ax, const ymm by) noexcept
    {
        return _mm256_permute2f128_pd(ax, by, 0x20); // == ab
    }

    static inline ymm cathi(const ymm ax, const ymm by) noexcept force_inline;
    static inline ymm cathi(const ymm ax, const ymm by) noexcept
    {
        return _mm256_permute2f128_pd(ax, by, 0x31); // == xy
    }

    static inline ymm swaplohi(const ymm ab) noexcept force_inline;
    static inline ymm swaplohi(const ymm ab) noexcept
    {
        return _mm256_permute2f128_pd(ab, ab, 0x01); // == ba
    }

    template <int s> static inline ymm getwp2(const_complex_vector W, const int p) noexcept force_inline;
    template <int s> static inline ymm getwp2(const_complex_vector W, const int p) noexcept
    {
        const int sp = s*p;
        return cmplx2(W[sp], W[sp+s]);
    }

    template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p) noexcept force_inline;
    template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p) noexcept
    {
        const int sp = s*p;
        return cnjpz2(cmplx2(W[sp], W[sp+s]));
    }

    static inline xmm getlo(const ymm a_b) noexcept force_inline;
    static inline xmm getlo(const ymm a_b) noexcept
    {
        return _mm256_castpd256_pd128(a_b); // == a
    }
    static inline xmm gethi(const ymm a_b) noexcept force_inline;
    static inline xmm gethi(const ymm a_b) noexcept
    {
        return _mm256_extractf128_pd(a_b, 1); // == b
    }

    template <int s> static inline ymm getpz3(const_complex_vector z) noexcept force_inline;
    template <int s> static inline ymm getpz3(const_complex_vector z) noexcept
    {
        return cmplx2(z[0], z[s]);
    }

    template <int s> static inline void setpz3(complex_vector z, const ymm x) noexcept force_inline3;
    template <int s> static inline void setpz3(complex_vector z, const ymm x) noexcept
    {
        setpz(z[0], getlo(x));
        setpz(z[s], gethi(x));
    }

    static inline void* simd_malloc(const size_t n) { return _mm_malloc(n, 32); }
    static inline void simd_free(void* p) { _mm_free(p); }

} // namespace OTFFT_MISC

#else
//=============================================================================
// AVX Emulation
//=============================================================================

namespace OTFFT_MISC {

    struct ymm { xmm lo, hi; };

    static inline void zeroupper() noexcept {}

    static inline ymm cmplx2(const double& a, const double& b, const double& c, const double &d) noexcept force_inline;
    static inline ymm cmplx2(const double& a, const double& b, const double& c, const double &d) noexcept
    {
        const ymm y = { cmplx(a, b), cmplx(c, d) };
        return y;
    }

    static inline ymm cmplx2(const complex_t& a, const complex_t& b) noexcept force_inline;
    static inline ymm cmplx2(const complex_t& a, const complex_t& b) noexcept
    {
        const ymm y = { getpz(a), getpz(b) };
        return y;
    }

    static inline ymm getpz2(const_complex_vector z) noexcept force_inline;
    static inline ymm getpz2(const_complex_vector z) noexcept
    {
        const ymm y = { getpz(z[0]), getpz(z[1]) };
        return y;
    }

    static inline void setpz2(complex_vector z, const ymm& y) noexcept force_inline3;
    static inline void setpz2(complex_vector z, const ymm& y) noexcept
    {
        setpz(z[0], y.lo);
        setpz(z[1], y.hi);
    }

    static inline ymm cnjpz2(const ymm& xy) noexcept force_inline;
    static inline ymm cnjpz2(const ymm& xy) noexcept
    {
        const ymm y = { cnjpz(xy.lo), cnjpz(xy.hi) };
        return y;
    }
    static inline ymm jxpz2(const ymm& xy) noexcept force_inline;
    static inline ymm jxpz2(const ymm& xy) noexcept
    {
        const ymm y = { jxpz(xy.lo), jxpz(xy.hi) };
        return y;
    }

    static inline ymm addpz2(const ymm& a, const ymm& b) noexcept force_inline;
    static inline ymm addpz2(const ymm& a, const ymm& b) noexcept
    {
        const ymm y = { addpz(a.lo, b.lo), addpz(a.hi, b.hi) };
        return y;
    }
    static inline ymm subpz2(const ymm& a, const ymm& b) noexcept force_inline;
    static inline ymm subpz2(const ymm& a, const ymm& b) noexcept
    {
        const ymm y = { subpz(a.lo, b.lo), subpz(a.hi, b.hi) };
        return y;
    }
    static inline ymm mulpd2(const ymm& a, const ymm& b) noexcept force_inline;
    static inline ymm mulpd2(const ymm& a, const ymm& b) noexcept
    {
        const ymm y = { mulpd(a.lo, b.lo), mulpd(a.hi, b.hi) };
        return y;
    }
    static inline ymm divpd2(const ymm& a, const ymm& b) noexcept force_inline;
    static inline ymm divpd2(const ymm& a, const ymm& b) noexcept
    {
        const ymm y = { divpd(a.lo, b.lo), divpd(a.hi, b.hi) };
        return y;
    }

    static inline ymm mulpz2(const ymm& a, const ymm& b) noexcept force_inline;
    static inline ymm mulpz2(const ymm& a, const ymm& b) noexcept
    {
        const ymm y = { mulpz(a.lo, b.lo), mulpz(a.hi, b.hi) };
        return y;
    }

    static inline ymm divpz2(const ymm& a, const ymm& b) noexcept force_inline;
    static inline ymm divpz2(const ymm& a, const ymm& b) noexcept
    {
        const ymm y = { divpz(a.lo, b.lo), divpz(a.hi, b.hi) };
        return y;
    }

    template <int N, int mode> static inline ymm scalepz2(const ymm z) noexcept force_inline;
    template <int N, int mode> static inline ymm scalepz2(const ymm z) noexcept
    {
        constexpr double scale =
            mode == scale_1       ? 1.0           :
            mode == scale_unitary ? 1.0/mysqrt(N) :
            mode == scale_length  ? 1.0/N         : 0.0;
        constexpr xmm sv  = { scale, scale };
        constexpr ymm sv2 = { sv, sv };
        return mode == scale_1 ? z : mulpd2(sv2, z);
    }

    static inline ymm v8xpz2(const ymm& xy) noexcept force_inline;
    static inline ymm v8xpz2(const ymm& xy) noexcept
    {
        const ymm y = { v8xpz(xy.lo), v8xpz(xy.hi) };
        return y;
    }

    static inline ymm w8xpz2(const ymm& xy) noexcept force_inline;
    static inline ymm w8xpz2(const ymm& xy) noexcept
    {
        const ymm y = { w8xpz(xy.lo), w8xpz(xy.hi) };
        return y;
    }

    static inline ymm h1xpz2(const ymm& xy) noexcept force_inline;
    static inline ymm h1xpz2(const ymm& xy) noexcept
    {
        const ymm y = { h1xpz(xy.lo), h1xpz(xy.hi) };
        return y;
    }

    static inline ymm h3xpz2(const ymm& xy) noexcept force_inline;
    static inline ymm h3xpz2(const ymm& xy) noexcept
    {
        const ymm y = { h3xpz(xy.lo), h3xpz(xy.hi) };
        return y;
    }

    static inline ymm hfxpz2(const ymm& xy) noexcept force_inline;
    static inline ymm hfxpz2(const ymm& xy) noexcept
    {
        const ymm y = { hfxpz(xy.lo), hfxpz(xy.hi) };
        return y;
    }

    static inline ymm hdxpz2(const ymm& xy) noexcept force_inline;
    static inline ymm hdxpz2(const ymm& xy) noexcept
    {
        const ymm y = { hdxpz(xy.lo), hdxpz(xy.hi) };
        return y;
    }

    static inline ymm duppz2(const xmm x) noexcept force_inline;
    static inline ymm duppz2(const xmm x) noexcept
    {
        const ymm y = { x, x }; return y;
    }

    static inline ymm duppz3(const complex_t& z) noexcept force_inline;
    static inline ymm duppz3(const complex_t& z) noexcept
    {
        const xmm x = getpz(z);
        const ymm y = { x, x };
        return y;
    }

    static inline ymm cat(const xmm& a, const xmm& b) noexcept force_inline;
    static inline ymm cat(const xmm& a, const xmm& b) noexcept
    {
        const ymm y = { a, b };
        return y;
    }

    static inline ymm catlo(const ymm& ax, const ymm& by) noexcept force_inline;
    static inline ymm catlo(const ymm& ax, const ymm& by) noexcept
    {
        const ymm ab = { ax.lo, by.lo };
        return ab;
    }

    static inline ymm cathi(const ymm ax, const ymm by) noexcept force_inline;
    static inline ymm cathi(const ymm ax, const ymm by) noexcept
    {
        const ymm xy = { ax.hi, by.hi };
        return xy;
    }

    static inline ymm swaplohi(const ymm ab) noexcept force_inline;
    static inline ymm swaplohi(const ymm ab) noexcept
    {
        const ymm xy = { ab.hi, ab.lo };
        return xy;
    }

    template <int s> static inline ymm getwp2(const_complex_vector W, const int p) noexcept force_inline;
    template <int s> static inline ymm getwp2(const_complex_vector W, const int p) noexcept
    {
        const int sp = s*p;
        return cmplx2(W[sp], W[sp+s]);
    }

    template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p) noexcept force_inline;
    template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p) noexcept
    {
        const int sp = s*p;
        return cnjpz2(cmplx2(W[sp], W[sp+s]));
    }

    static inline xmm getlo(const ymm& a_b) noexcept force_inline;
    static inline xmm getlo(const ymm& a_b) noexcept { return a_b.lo; }
    static inline xmm gethi(const ymm& a_b) noexcept force_inline;
    static inline xmm gethi(const ymm& a_b) noexcept { return a_b.hi; }

    template <int s> static inline ymm getpz3(const_complex_vector z) noexcept force_inline;
    template <int s> static inline ymm getpz3(const_complex_vector z) noexcept
    {
        return cmplx2(z[0], z[s]);
    }

    template <int s> static inline void setpz3(complex_vector z, const ymm& y) noexcept force_inline3;
    template <int s> static inline void setpz3(complex_vector z, const ymm& y) noexcept
    {
        setpz(z[0], getlo(y));
        setpz(z[s], gethi(y));
    }

} // namespace OTFFT_MISC

#endif // __AVX__

//=============================================================================
// Aligned Memory Allocator
//=============================================================================

namespace OTFFT_MISC {

    template <class T> struct simd_array
    {
        T* p;

        simd_array() noexcept : p(0) {}
        simd_array(int n) : p((T*) simd_malloc(n*sizeof(T)))
        {
            if (p == 0) throw std::bad_alloc();
        }

        ~simd_array() { if (p) simd_free(p); }

        void setup(int n)
        {
            if (p) simd_free(p);
            p = (T*) simd_malloc(n*sizeof(T));
            if (p == 0) throw std::bad_alloc();
        }

        void destroy() { if (p) simd_free(p); p = 0; }

        T& operator[](int i) noexcept { return p[i]; }
        const T& operator[](int i) const noexcept { return p[i]; }
        T* operator&() const noexcept { return p; }
    };

} // namespace OTFFT_MISC

//=============================================================================
#endif // otfft_misc_h
