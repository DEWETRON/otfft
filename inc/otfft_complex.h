/******************************************************************************
*  OTFFT Complex & Memory Allocator Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_complex_h
#define otfft_complex_h

#ifdef _MSC_VER
#if _MSC_VER >= 1900
#define VC_CONSTEXPR 1
#else
#error "This compiler is not supported."
#endif
#endif // _MSC_VER

#if __cplusplus >= 201103L || defined(VC_CONSTEXPR)
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

#if __GNUC__ >= 3
#define force_inline  __attribute__((const,always_inline))
#define force_inline2 __attribute__((pure,always_inline))
#define force_inline3 __attribute__((always_inline))
//#define force_inline  __attribute__((const))
//#define force_inline2 __attribute__((pure))
//#define force_inline3
//#define force_inline
//#define force_inline2
//#define force_inline3
#else
#define force_inline
#define force_inline2
#define force_inline3
#endif

#ifdef _MSC_VER

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
#define __FMA__  1
#endif
#ifdef __AVX512F__
#define __SSE2__ 1
#define __SSE3__ 1
#define __FMA__  1
#endif

#endif // _MSC_VER

//=============================================================================
// User Defined Complex Number Class
//=============================================================================

#include <complex>

namespace OTFFT {

struct complex_t
{
    double Re, Im;

    complex_t() NOEXCEPT : Re(0), Im(0) {}
    complex_t(double x) NOEXCEPT : Re(x), Im(0) {}
    complex_t(double x, double y) NOEXCEPT : Re(x), Im(y) {}
    complex_t(const std::complex<double>& z) NOEXCEPT : Re(z.real()), Im(z.imag()) {}
    operator std::complex<double>() const { return std::complex<double>(Re, Im); }

    complex_t& operator+=(const complex_t& z) NOEXCEPT
    {
        Re += z.Re;
        Im += z.Im;
        return *this;
    }

    complex_t& operator-=(const complex_t& z) NOEXCEPT
    {
        Re -= z.Re;
        Im -= z.Im;
        return *this;
    }

    complex_t& operator*=(const double& x) NOEXCEPT
    {
        Re *= x;
        Im *= x;
        return *this;
    }

    complex_t& operator/=(const double& x) NOEXCEPT
    {
        Re /= x;
        Im /= x;
        return *this;
    }

    complex_t& operator*=(const complex_t& z) NOEXCEPT
    {
        const double tmp = Re*z.Re - Im*z.Im;
        Im = Re*z.Im + Im*z.Re;
        Re = tmp;
        return *this;
    }
};

#if 1
typedef double* __restrict const double_vector;
typedef const double* __restrict const const_double_vector;
typedef complex_t* __restrict const complex_vector;
typedef const complex_t* __restrict const const_complex_vector;
#else
typedef double* const double_vector;
typedef const double* const const_double_vector;
typedef complex_t* const complex_vector;
typedef const complex_t* const const_complex_vector;
#endif

static inline double Re(const complex_t& z) NOEXCEPT force_inline;
static inline double Re(const complex_t& z) NOEXCEPT { return z.Re; }
static inline double Im(const complex_t& z) NOEXCEPT force_inline;
static inline double Im(const complex_t& z) NOEXCEPT { return z.Im; }

static inline double norm(const complex_t& z) NOEXCEPT force_inline;
static inline double norm(const complex_t& z) NOEXCEPT
{
    return z.Re*z.Re + z.Im*z.Im;
}
static inline complex_t conj(const complex_t& z) NOEXCEPT force_inline;
static inline complex_t conj(const complex_t& z) NOEXCEPT
{
    return complex_t(z.Re, -z.Im);
}
static inline double abs(const complex_t& z) NOEXCEPT force_inline;
static inline double abs(const complex_t& z) NOEXCEPT
{
    return sqrt(norm(z));
}
static inline double arg(const complex_t& z) NOEXCEPT force_inline;
static inline double arg(const complex_t& z) NOEXCEPT
{
    return atan2(z.Im, z.Re);
}
static inline complex_t proj(const complex_t& z) NOEXCEPT force_inline;
static inline complex_t proj(const complex_t& z) NOEXCEPT
{
    const double den(norm(z) + double(1.0));
    return complex_t(double(2.0) * z.Re / den, double(2.0) * z.Im / den);
}
static inline complex_t polar(const double rho, const double theta) NOEXCEPT force_inline;
static inline complex_t polar(const double rho, const double theta) NOEXCEPT
{
    return complex_t(rho * cos(theta), rho * sin(theta));
}
static inline complex_t jx(const complex_t& z) NOEXCEPT force_inline;
static inline complex_t jx(const complex_t& z) NOEXCEPT
{
    return complex_t(-z.Im, z.Re);
}
static inline complex_t neg(const complex_t& z) NOEXCEPT force_inline;
static inline complex_t neg(const complex_t& z) NOEXCEPT
{
    return complex_t(-z.Re, -z.Im);
}
static inline complex_t mjx(const complex_t& z) NOEXCEPT force_inline;
static inline complex_t mjx(const complex_t& z) NOEXCEPT
{
    return complex_t(z.Im, -z.Re);
}
static inline complex_t operator+(const complex_t& a, const complex_t& b) NOEXCEPT force_inline;
static inline complex_t operator+(const complex_t& a, const complex_t& b) NOEXCEPT
{
    return complex_t(a.Re + b.Re, a.Im + b.Im);
}
static inline complex_t operator-(const complex_t& a, const complex_t& b) NOEXCEPT force_inline;
static inline complex_t operator-(const complex_t& a, const complex_t& b) NOEXCEPT
{
    return complex_t(a.Re - b.Re, a.Im - b.Im);
}
static inline complex_t operator*(const double& a, const complex_t& b) NOEXCEPT force_inline;
static inline complex_t operator*(const double& a, const complex_t& b) NOEXCEPT
{
    return complex_t(a*b.Re, a*b.Im);
}
static inline complex_t operator*(const complex_t& a, const complex_t& b) NOEXCEPT force_inline;
static inline complex_t operator*(const complex_t& a, const complex_t& b) NOEXCEPT
{
    return complex_t(a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re);
}
static inline complex_t operator/(const complex_t& a, const double& b) NOEXCEPT force_inline;
static inline complex_t operator/(const complex_t& a, const double& b) NOEXCEPT
{
    return complex_t(a.Re/b, a.Im/b);
}
static inline complex_t operator/(const complex_t& a, const complex_t& b) NOEXCEPT force_inline;
static inline complex_t operator/(const complex_t& a, const complex_t& b) NOEXCEPT
{
    const double b2 = b.Re*b.Re + b.Im*b.Im;
    return (a * conj(b)) / b2;
}

static inline complex_t expj(const double& theta) NOEXCEPT force_inline;
static inline complex_t expj(const double& theta) NOEXCEPT
{
    //return complex_t(cos(theta), sin(theta));
    return complex_t(std::polar(1.0, theta));
}

} // namespace OTFFT

//=============================================================================
// Aligned Memory Allocator
//=============================================================================

#include <new>
#include <immintrin.h>

#ifdef __MINGW32__
#include <malloc.h>
#endif

namespace OTFFT {

#ifdef __SSE2__
#ifndef __AVX__
static inline void* simd_malloc(const size_t n) { return _mm_malloc(n, 16); }
#endif
#else
static inline void* simd_malloc(const size_t n) { return malloc(n); }
#endif

#ifdef __AVX__
#ifdef __AVX512F__
static inline void* simd_malloc(const size_t n) { return _mm_malloc(n, 64); }
#else
static inline void* simd_malloc(const size_t n) { return _mm_malloc(n, 32); }
#endif
#endif

static inline void simd_free(void* p) { _mm_free(p); }

template <class T> struct simd_array
{
    T* p;

    simd_array() NOEXCEPT : p(0) {}
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

    T& operator[](int i) NOEXCEPT { return p[i]; }
    const T& operator[](int i) const NOEXCEPT { return p[i]; }
    T* operator&() const NOEXCEPT { return p; }
};

} // namespace OTFFT

//=============================================================================

#endif // otfft_complex_h
