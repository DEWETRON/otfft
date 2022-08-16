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

    static void init_Wr1(const int r, const int N, complex_vector W) noexcept
    {
       if (N < r) return;
       const int Nr = N/r;
       const double theta = -2*M_PI/N;
       if (N < OMP_THRESHOLD_W) {
           for (int p = 0; p < Nr; p++) {
               W[p] = expj(theta * p);
           }
       }
       else {
#pragma omp parallel for schedule(static)
           for (int p = 0; p < Nr; p++) {
               W[p] = expj(theta * p);
           }
       }
    }

    static void init_Wt(const int r, const int N, complex_vector W) noexcept
    {
       if (N < r) return;
       const int Nr = N/r;
       const double theta = -2*M_PI/N;
       if (N < OMP_THRESHOLD_W) {
           for (int p = 0; p < Nr; p++) {
               for (int k = 1; k < r; k++) {
                   W[p + (k-1)*Nr] = W[N + r*p + k] = expj(theta * k*p);
               }
           }
       }
       else {
#pragma omp parallel for schedule(static)
           for (int p = 0; p < Nr; p++) {
               for (int k = 1; k < r; k++) {
                   W[p + (k-1)*Nr] = W[N + r*p + k] = expj(theta * k*p);
               }
           }
       }
    }

    template <int r, int N, int k>
    const complex_t* twid(const_complex_vector W, int p)
    {
        constexpr int Nr = N/r;
        constexpr int d = (k-1)*Nr;
        return W + p + d;
    }
    
    template <int r, int N, int k>
    const complex_t* twidT(const_complex_vector W, int p)
    {
        constexpr int d = N + k;
        return W + r*p + d;
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

template <int N, int s>
static xmm modqpz(const_complex_vector W, const int p) noexcept
{
    constexpr int Nq = N/4;
    constexpr int log_Nq = mylog2(Nq);
    const int sp = s*p;
    const int q = sp >> log_Nq;
    const int r = sp & (Nq-1);
    const xmm x = getpz(W[r]);
    switch (q & 3) {
        case 0: return x;
        case 1: return mjxpz(x);
        case 2: return negpz(x);
        case 3: return jxpz(x);
    }
    return xmm();
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

template <int N, int s>
static xmm modqpz(const_complex_vector W, const int p) noexcept
{
    constexpr int Nq = N/4;
    constexpr int log_Nq = mylog2(Nq);
    const int sp = s*p;
    const int q = sp >> log_Nq;
    const int r = sp & (Nq-1);
    const xmm x = getpz(W[r]);
    switch (q & 3) {
        case 0: return x;
        case 1: return mjxpz(x);
        case 2: return negpz(x);
        case 3: return jxpz(x);
    }
    return xmm();
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

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(USE_INTRINSIC)
//=============================================================================
// AVX-512
//=============================================================================

#include <immintrin.h>

namespace OTFFT_MISC {

typedef __m512d zmm;

static inline zmm getpz4(const_complex_vector z) noexcept force_inline;
static inline zmm getpz4(const_complex_vector z) noexcept
{
#ifdef USE_UNALIGNED_MEMORY
    return _mm512_loadu_pd(&z->Re);
#else
    return _mm512_load_pd(&z->Re);
#endif
}

static inline void setpz4(complex_vector a, const zmm z) noexcept force_inline3;
static inline void setpz4(complex_vector a, const zmm z) noexcept
{
#ifdef USE_UNALIGNED_MEMORY
    _mm512_storeu_pd(&a->Re, z);
#else
    _mm512_store_pd(&a->Re, z);
#endif
}

static inline zmm cnjpz4(const zmm xy) noexcept force_inline;
static inline zmm cnjpz4(const zmm xy) noexcept
{
    constexpr zmm zm = { 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0 };
    return _mm512_xor_pd(zm, xy);
}
static inline zmm jxpz4(const zmm xy) noexcept force_inline;
static inline zmm jxpz4(const zmm xy) noexcept
{
    const zmm xmy = cnjpz4(xy);
    return _mm512_shuffle_pd(xmy, xmy, 0x55);
}

static inline zmm addpz4(const zmm a, const zmm b) noexcept force_inline;
static inline zmm addpz4(const zmm a, const zmm b) noexcept
{
    return _mm512_add_pd(a, b);
}
static inline zmm subpz4(const zmm a, const zmm b) noexcept force_inline;
static inline zmm subpz4(const zmm a, const zmm b) noexcept
{
    return _mm512_sub_pd(a, b);
}
static inline zmm mulpd4(const zmm a, const zmm b) noexcept force_inline;
static inline zmm mulpd4(const zmm a, const zmm b) noexcept
{
    return _mm512_mul_pd(a, b);
}
static inline zmm divpd4(const zmm a, const zmm b) noexcept force_inline;
static inline zmm divpd4(const zmm a, const zmm b) noexcept
{
    return _mm512_div_pd(a, b);
}

static inline zmm mulpz4(const zmm ab, const zmm xy) noexcept force_inline;
static inline zmm mulpz4(const zmm ab, const zmm xy) noexcept
{
    const zmm aa = _mm512_unpacklo_pd(ab, ab);
    const zmm bb = _mm512_unpackhi_pd(ab, ab);
    const zmm yx = _mm512_shuffle_pd(xy, xy, 0x55);
    return _mm512_fmaddsub_pd(aa, xy, _mm512_mul_pd(bb, yx));
}

template <int N, int mode> static inline zmm scalepz4(const zmm z) noexcept force_inline;
template <int N, int mode> static inline zmm scalepz4(const zmm z) noexcept
{
    constexpr double sc =
        mode == scale_1       ? 1.0           :
        mode == scale_unitary ? 1.0/mysqrt(N) :
        mode == scale_length  ? 1.0/N         : 0.0;
    constexpr zmm sv  = { sc, sc, sc, sc, sc, sc, sc, sc };
    return mode == scale_1 ? z : mulpd4(sv, z);
}

static inline zmm v8xpz4(const zmm xy) noexcept force_inline;
static inline zmm v8xpz4(const zmm xy) noexcept
{
    constexpr double r = M_SQRT1_2;
    constexpr zmm rr = { r, r, r, r, r, r, r, r };
    const zmm myx = jxpz4(xy);
    return _mm512_mul_pd(rr, _mm512_add_pd(xy, myx));
}

static inline zmm w8xpz4(const zmm xy) noexcept force_inline;
static inline zmm w8xpz4(const zmm xy) noexcept
{
    constexpr double r = M_SQRT1_2;
    constexpr zmm rr = { r, r, r, r, r, r, r, r };
    const zmm ymx = cnjpz4(_mm512_shuffle_pd(xy, xy, 0x55));
    return _mm512_mul_pd(rr, _mm512_add_pd(xy, ymx));
}

static inline zmm h1xpz4(const zmm xy) noexcept force_inline;
static inline zmm h1xpz4(const zmm xy) noexcept
{
    constexpr zmm h1 = { H1X, H1Y, H1X, H1Y, H1X, H1Y, H1X, H1Y };
    return mulpz4(h1, xy);
}

static inline zmm h3xpz4(const zmm xy) noexcept force_inline;
static inline zmm h3xpz4(const zmm xy) noexcept
{
    constexpr zmm h3 = { -H1Y, -H1X, -H1Y, -H1X, -H1Y, -H1X, -H1Y, -H1X };
    return mulpz4(h3, xy);
}

static inline zmm hfxpz4(const zmm xy) noexcept force_inline;
static inline zmm hfxpz4(const zmm xy) noexcept
{
    constexpr zmm hf = { H1X, -H1Y, H1X, -H1Y, H1X, -H1Y, H1X, -H1Y };
    return mulpz4(hf, xy);
}

static inline zmm hdxpz4(const zmm xy) noexcept force_inline;
static inline zmm hdxpz4(const zmm xy) noexcept
{
    constexpr zmm hd = { -H1Y, H1X, -H1Y, H1X, -H1Y, H1X, -H1Y, H1X };
    return mulpz4(hd, xy);
}

static inline zmm duppz4(const xmm x) noexcept force_inline;
static inline zmm duppz4(const xmm x) noexcept
{
    return _mm512_broadcast_f64x2(x);
}

static inline zmm duppz5(const complex_t& z) noexcept force_inline;
static inline zmm duppz5(const complex_t& z) noexcept
{
    return duppz4(getpz(z));
}

} // namespace OTFFT_MISC

#else
//=============================================================================
// AVX-512 Emulation
//=============================================================================

namespace OTFFT_MISC {

struct zmm { ymm lo, hi; };

static inline zmm getpz4(const_complex_vector a) noexcept force_inline;
static inline zmm getpz4(const_complex_vector a) noexcept
{
    const zmm z = { getpz2(&a[0]), getpz2(&a[2]) };
    return z;
}

static inline void setpz4(complex_vector a, const zmm& z) noexcept force_inline3;
static inline void setpz4(complex_vector a, const zmm& z) noexcept
{
    setpz2(&a[0], z.lo);
    setpz2(&a[2], z.hi);
}

static inline zmm cnjpz4(const zmm& xy) noexcept force_inline;
static inline zmm cnjpz4(const zmm& xy) noexcept
{
    const zmm z = { cnjpz2(xy.lo), cnjpz2(xy.hi) };
    return z;
}
static inline zmm jxpz4(const zmm& xy) noexcept force_inline;
static inline zmm jxpz4(const zmm& xy) noexcept
{
    const zmm z = { jxpz2(xy.lo), jxpz2(xy.hi) };
    return z;
}

static inline zmm addpz4(const zmm& a, const zmm& b) noexcept force_inline;
static inline zmm addpz4(const zmm& a, const zmm& b) noexcept
{
    const zmm z = { addpz2(a.lo, b.lo), addpz2(a.hi, b.hi) };
    return z;
}
static inline zmm subpz4(const zmm& a, const zmm& b) noexcept force_inline;
static inline zmm subpz4(const zmm& a, const zmm& b) noexcept
{
    const zmm z = { subpz2(a.lo, b.lo), subpz2(a.hi, b.hi) };
    return z;
}
static inline zmm mulpd4(const zmm& a, const zmm& b) noexcept force_inline;
static inline zmm mulpd4(const zmm& a, const zmm& b) noexcept
{
    const zmm z = { mulpd2(a.lo, b.lo), mulpd2(a.hi, b.hi) };
    return z;
}
static inline zmm divpd4(const zmm& a, const zmm& b) noexcept force_inline;
static inline zmm divpd4(const zmm& a, const zmm& b) noexcept
{
    const zmm z = { divpd2(a.lo, b.lo), divpd2(a.hi, b.hi) };
    return z;
}

static inline zmm mulpz4(const zmm& a, const zmm& b) noexcept force_inline;
static inline zmm mulpz4(const zmm& a, const zmm& b) noexcept
{
    const zmm z = { mulpz2(a.lo, b.lo), mulpz2(a.hi, b.hi) };
    return z;
}

template <int N, int mode> static inline zmm scalepz4(const zmm z) noexcept force_inline;
template <int N, int mode> static inline zmm scalepz4(const zmm z) noexcept
{
    constexpr double scale =
        mode == scale_1       ? 1.0           :
        mode == scale_unitary ? 1.0/mysqrt(N) :
        mode == scale_length  ? 1.0/N         : 0.0;
    constexpr ymm sv  = { scale, scale, scale, scale };
    constexpr zmm sv4 = { sv, sv };
    return mode == scale_1 ? z : mulpd4(sv4, z);
}

static inline zmm v8xpz4(const zmm& xy) noexcept force_inline;
static inline zmm v8xpz4(const zmm& xy) noexcept
{
    const zmm z = { v8xpz2(xy.lo), v8xpz2(xy.hi) };
    return z;
}

static inline zmm w8xpz4(const zmm& xy) noexcept force_inline;
static inline zmm w8xpz4(const zmm& xy) noexcept
{
    const zmm z = { w8xpz2(xy.lo), w8xpz2(xy.hi) };
    return z;
}

static inline zmm h1xpz4(const zmm& xy) noexcept force_inline;
static inline zmm h1xpz4(const zmm& xy) noexcept
{
    const zmm z = { h1xpz2(xy.lo), h1xpz2(xy.hi) };
    return z;
}

static inline zmm h3xpz4(const zmm& xy) noexcept force_inline;
static inline zmm h3xpz4(const zmm& xy) noexcept
{
    const zmm z = { h3xpz2(xy.lo), h3xpz2(xy.hi) };
    return z;
}

static inline zmm hfxpz4(const zmm& xy) noexcept force_inline;
static inline zmm hfxpz4(const zmm& xy) noexcept
{
    const zmm z = { hfxpz2(xy.lo), hfxpz2(xy.hi) };
    return z;
}

static inline zmm hdxpz4(const zmm& xy) noexcept force_inline;
static inline zmm hdxpz4(const zmm& xy) noexcept
{
    const zmm z = { hdxpz2(xy.lo), hdxpz2(xy.hi) };
    return z;
}

static inline zmm duppz4(const xmm x) noexcept force_inline;
static inline zmm duppz4(const xmm x) noexcept
{
    const ymm y = duppz2(x);
    const zmm z = { y, y };
    return z;
}

static inline zmm duppz5(const complex_t& z) noexcept force_inline;
static inline zmm duppz5(const complex_t& z) noexcept
{
    return duppz4(getpz(z));
}

} // namespace OTFFT_MISC

#endif // __AVX512F__ && __AVX512DQ__

//=============================================================================
// 512 bit Vector Emulation
//=============================================================================

namespace OTFFT_MISC {

struct emm { ymm lo, hi; };

static inline emm getez4(const_complex_vector a) noexcept force_inline;
static inline emm getez4(const_complex_vector a) noexcept
{
    const emm z = { getpz2(&a[0]), getpz2(&a[2]) };
    return z;
}

static inline void setez4(complex_vector a, const emm& z) noexcept force_inline3;
static inline void setez4(complex_vector a, const emm& z) noexcept
{
    setpz2(&a[0], z.lo);
    setpz2(&a[2], z.hi);
}

static inline emm cnjez4(const emm& xy) noexcept force_inline;
static inline emm cnjez4(const emm& xy) noexcept
{
    const emm z = { cnjpz2(xy.lo), cnjpz2(xy.hi) };
    return z;
}
static inline emm jxez4(const emm& xy) noexcept force_inline;
static inline emm jxez4(const emm& xy) noexcept
{
    const emm z = { jxpz2(xy.lo), jxpz2(xy.hi) };
    return z;
}

static inline emm addez4(const emm& a, const emm& b) noexcept force_inline;
static inline emm addez4(const emm& a, const emm& b) noexcept
{
    const emm z = { addpz2(a.lo, b.lo), addpz2(a.hi, b.hi) };
    return z;
}
static inline emm subez4(const emm& a, const emm& b) noexcept force_inline;
static inline emm subez4(const emm& a, const emm& b) noexcept
{
    const emm z = { subpz2(a.lo, b.lo), subpz2(a.hi, b.hi) };
    return z;
}
static inline emm muled4(const emm& a, const emm& b) noexcept force_inline;
static inline emm muled4(const emm& a, const emm& b) noexcept
{
    const emm z = { mulpd2(a.lo, b.lo), mulpd2(a.hi, b.hi) };
    return z;
}
static inline emm dived4(const emm& a, const emm& b) noexcept force_inline;
static inline emm dived4(const emm& a, const emm& b) noexcept
{
    const emm z = { divpd2(a.lo, b.lo), divpd2(a.hi, b.hi) };
    return z;
}

static inline emm mulez4(const emm& a, const emm& b) noexcept force_inline;
static inline emm mulez4(const emm& a, const emm& b) noexcept
{
    const emm z = { mulpz2(a.lo, b.lo), mulpz2(a.hi, b.hi) };
    return z;
}

template <int N, int mode> static inline emm scaleez4(const emm z) noexcept force_inline;
template <int N, int mode> static inline emm scaleez4(const emm z) noexcept
{
    constexpr double scale =
        mode == scale_1       ? 1.0           :
        mode == scale_unitary ? 1.0/mysqrt(N) :
        mode == scale_length  ? 1.0/N         : 0.0;
    constexpr ymm sv  = { scale, scale, scale, scale };
    constexpr emm sv4 = { sv, sv };
    return mode == scale_1 ? z : muled4(sv4, z);
}

static inline emm v8xez4(const emm& xy) noexcept force_inline;
static inline emm v8xez4(const emm& xy) noexcept
{
    const emm z = { v8xpz2(xy.lo), v8xpz2(xy.hi) };
    return z;
}

static inline emm w8xez4(const emm& xy) noexcept force_inline;
static inline emm w8xez4(const emm& xy) noexcept
{
    const emm z = { w8xpz2(xy.lo), w8xpz2(xy.hi) };
    return z;
}

static inline emm h1xez4(const emm& xy) noexcept force_inline;
static inline emm h1xez4(const emm& xy) noexcept
{
    const emm z = { h1xpz2(xy.lo), h1xpz2(xy.hi) };
    return z;
}

static inline emm h3xez4(const emm& xy) noexcept force_inline;
static inline emm h3xez4(const emm& xy) noexcept
{
    const emm z = { h3xpz2(xy.lo), h3xpz2(xy.hi) };
    return z;
}

static inline emm hfxez4(const emm& xy) noexcept force_inline;
static inline emm hfxez4(const emm& xy) noexcept
{
    const emm z = { hfxpz2(xy.lo), hfxpz2(xy.hi) };
    return z;
}

static inline emm hdxez4(const emm& xy) noexcept force_inline;
static inline emm hdxez4(const emm& xy) noexcept
{
    const emm z = { hdxpz2(xy.lo), hdxpz2(xy.hi) };
    return z;
}

static inline emm dupez4(const xmm x) noexcept force_inline;
static inline emm dupez4(const xmm x) noexcept
{
    const ymm y = duppz2(x);
    const emm z = { y, y };
    return z;
}

static inline emm dupez5(const complex_t& z) noexcept force_inline;
static inline emm dupez5(const complex_t& z) noexcept
{
    return dupez4(getpz(z));
}

} // namespace OTFFT_MISC

//=============================================================================
// 1024 bit Vector Emulation
//=============================================================================

namespace OTFFT_MISC {

struct amm { zmm lo, hi; };

static inline amm getpz8(const_complex_vector a) noexcept force_inline;
static inline amm getpz8(const_complex_vector a) noexcept
{
    const amm z = { getpz4(&a[0]), getpz4(&a[4]) };
    return z;
}

static inline void setpz8(complex_vector a, const amm& z) noexcept force_inline3;
static inline void setpz8(complex_vector a, const amm& z) noexcept
{
    setpz4(&a[0], z.lo);
    setpz4(&a[4], z.hi);
}

static inline amm cnjpz8(const amm& xy) noexcept force_inline;
static inline amm cnjpz8(const amm& xy) noexcept
{
    const amm z = { cnjpz4(xy.lo), cnjpz4(xy.hi) };
    return z;
}
static inline amm jxpz8(const amm& xy) noexcept force_inline;
static inline amm jxpz8(const amm& xy) noexcept
{
    const amm z = { jxpz4(xy.lo), jxpz4(xy.hi) };
    return z;
}

static inline amm addpz8(const amm& a, const amm& b) noexcept force_inline;
static inline amm addpz8(const amm& a, const amm& b) noexcept
{
    const amm z = { addpz4(a.lo, b.lo), addpz4(a.hi, b.hi) };
    return z;
}
static inline amm subpz8(const amm& a, const amm& b) noexcept force_inline;
static inline amm subpz8(const amm& a, const amm& b) noexcept
{
    const amm z = { subpz4(a.lo, b.lo), subpz4(a.hi, b.hi) };
    return z;
}
static inline amm mulpd8(const amm& a, const amm& b) noexcept force_inline;
static inline amm mulpd8(const amm& a, const amm& b) noexcept
{
    const amm z = { mulpd4(a.lo, b.lo), mulpd4(a.hi, b.hi) };
    return z;
}
static inline amm divpd8(const amm& a, const amm& b) noexcept force_inline;
static inline amm divpd8(const amm& a, const amm& b) noexcept
{
    const amm z = { divpd4(a.lo, b.lo), divpd4(a.hi, b.hi) };
    return z;
}

static inline amm mulpz8(const amm& a, const amm& b) noexcept force_inline;
static inline amm mulpz8(const amm& a, const amm& b) noexcept
{
    const amm z = { mulpz4(a.lo, b.lo), mulpz4(a.hi, b.hi) };
    return z;
}

template <int N, int mode> static inline amm scalepz8(const amm z) noexcept force_inline;
template <int N, int mode> static inline amm scalepz8(const amm z) noexcept
{
    constexpr double sc =
        mode == scale_1       ? 1.0           :
        mode == scale_unitary ? 1.0/mysqrt(N) :
        mode == scale_length  ? 1.0/N         : 0.0;
    constexpr zmm sv  = { sc, sc, sc, sc, sc, sc, sc, sc };
    constexpr amm sv8 = { sv, sv };
    return mode == scale_1 ? z : mulpd8(sv8, z);
}

static inline amm v8xpz8(const amm& xy) noexcept force_inline;
static inline amm v8xpz8(const amm& xy) noexcept
{
    const amm z = { v8xpz4(xy.lo), v8xpz4(xy.hi) };
    return z;
}

static inline amm w8xpz8(const amm& xy) noexcept force_inline;
static inline amm w8xpz8(const amm& xy) noexcept
{
    const amm z = { w8xpz4(xy.lo), w8xpz4(xy.hi) };
    return z;
}

static inline amm h1xpz8(const amm& xy) noexcept force_inline;
static inline amm h1xpz8(const amm& xy) noexcept
{
    const amm z = { h1xpz4(xy.lo), h1xpz4(xy.hi) };
    return z;
}

static inline amm h3xpz8(const amm& xy) noexcept force_inline;
static inline amm h3xpz8(const amm& xy) noexcept
{
    const amm z = { h3xpz4(xy.lo), h3xpz4(xy.hi) };
    return z;
}

static inline amm hfxpz8(const amm& xy) noexcept force_inline;
static inline amm hfxpz8(const amm& xy) noexcept
{
    const amm z = { hfxpz4(xy.lo), hfxpz4(xy.hi) };
    return z;
}

static inline amm hdxpz8(const amm& xy) noexcept force_inline;
static inline amm hdxpz8(const amm& xy) noexcept
{
    const amm z = { hdxpz4(xy.lo), hdxpz4(xy.hi) };
    return z;
}

static inline amm duppz8(const xmm x) noexcept force_inline;
static inline amm duppz8(const xmm x) noexcept
{
    const zmm y = duppz4(x);
    const amm z = { y, y };
    return z;
}

static inline amm duppz9(const complex_t& z) noexcept force_inline;
static inline amm duppz9(const complex_t& z) noexcept
{
    return duppz8(getpz(z));
}

} // namespace OTFFT_MISC

//=============================================================================
// 2048 bit Vector Emulation
//=============================================================================

namespace OTFFT_MISC {

struct bmm { amm lo, hi; };

static inline bmm getpz16(const_complex_vector a) noexcept force_inline;
static inline bmm getpz16(const_complex_vector a) noexcept
{
    const bmm z = { getpz8(&a[0]), getpz8(&a[8]) };
    return z;
}

static inline void setpz16(complex_vector a, const bmm& z) noexcept force_inline3;
static inline void setpz16(complex_vector a, const bmm& z) noexcept
{
    setpz8(&a[0], z.lo);
    setpz8(&a[8], z.hi);
}

static inline bmm cnjpz16(const bmm& xy) noexcept force_inline;
static inline bmm cnjpz16(const bmm& xy) noexcept
{
    const bmm z = { cnjpz8(xy.lo), cnjpz8(xy.hi) };
    return z;
}
static inline bmm jxpz16(const bmm& xy) noexcept force_inline;
static inline bmm jxpz16(const bmm& xy) noexcept
{
    const bmm z = { jxpz8(xy.lo), jxpz8(xy.hi) };
    return z;
}

static inline bmm addpz16(const bmm& a, const bmm& b) noexcept force_inline;
static inline bmm addpz16(const bmm& a, const bmm& b) noexcept
{
    const bmm z = { addpz8(a.lo, b.lo), addpz8(a.hi, b.hi) };
    return z;
}
static inline bmm subpz16(const bmm& a, const bmm& b) noexcept force_inline;
static inline bmm subpz16(const bmm& a, const bmm& b) noexcept
{
    const bmm z = { subpz8(a.lo, b.lo), subpz8(a.hi, b.hi) };
    return z;
}
static inline bmm mulpd16(const bmm& a, const bmm& b) noexcept force_inline;
static inline bmm mulpd16(const bmm& a, const bmm& b) noexcept
{
    const bmm z = { mulpd8(a.lo, b.lo), mulpd8(a.hi, b.hi) };
    return z;
}
static inline bmm divpd16(const bmm& a, const bmm& b) noexcept force_inline;
static inline bmm divpd16(const bmm& a, const bmm& b) noexcept
{
    const bmm z = { divpd8(a.lo, b.lo), divpd8(a.hi, b.hi) };
    return z;
}

static inline bmm mulpz16(const bmm& a, const bmm& b) noexcept force_inline;
static inline bmm mulpz16(const bmm& a, const bmm& b) noexcept
{
    const bmm z = { mulpz8(a.lo, b.lo), mulpz8(a.hi, b.hi) };
    return z;
}

template <int N, int mode> static inline bmm scalepz16(const bmm z) noexcept force_inline;
template <int N, int mode> static inline bmm scalepz16(const bmm z) noexcept
{
    constexpr double sc =
        mode == scale_1       ? 1.0           :
        mode == scale_unitary ? 1.0/mysqrt(N) :
        mode == scale_length  ? 1.0/N         : 0.0;
    constexpr amm sv  = { sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc };
    constexpr bmm sv16 = { sv, sv };
    return mode == scale_1 ? z : mulpd16(sv16, z);
}

static inline bmm v8xpz16(const bmm& xy) noexcept force_inline;
static inline bmm v8xpz16(const bmm& xy) noexcept
{
    const bmm z = { v8xpz8(xy.lo), v8xpz8(xy.hi) };
    return z;
}

static inline bmm w8xpz16(const bmm& xy) noexcept force_inline;
static inline bmm w8xpz16(const bmm& xy) noexcept
{
    const bmm z = { w8xpz8(xy.lo), w8xpz8(xy.hi) };
    return z;
}

static inline bmm h1xpz16(const bmm& xy) noexcept force_inline;
static inline bmm h1xpz16(const bmm& xy) noexcept
{
    const bmm z = { h1xpz8(xy.lo), h1xpz8(xy.hi) };
    return z;
}

static inline bmm h3xpz16(const bmm& xy) noexcept force_inline;
static inline bmm h3xpz16(const bmm& xy) noexcept
{
    const bmm z = { h3xpz8(xy.lo), h3xpz8(xy.hi) };
    return z;
}

static inline bmm hfxpz16(const bmm& xy) noexcept force_inline;
static inline bmm hfxpz16(const bmm& xy) noexcept
{
    const bmm z = { hfxpz8(xy.lo), hfxpz8(xy.hi) };
    return z;
}

static inline bmm hdxpz16(const bmm& xy) noexcept force_inline;
static inline bmm hdxpz16(const bmm& xy) noexcept
{
    const bmm z = { hdxpz8(xy.lo), hdxpz8(xy.hi) };
    return z;
}

static inline bmm duppz16(const xmm x) noexcept force_inline;
static inline bmm duppz16(const xmm x) noexcept
{
    const amm y = duppz8(x);
    const bmm z = { y, y };
    return z;
}

static inline bmm duppz17(const complex_t& z) noexcept force_inline;
static inline bmm duppz17(const complex_t& z) noexcept
{
    return duppz16(getpz(z));
}

} // namespace OTFFT_MISC

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
