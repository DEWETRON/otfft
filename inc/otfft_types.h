// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

//=============================================================================
// Customizing Parameter
//=============================================================================

#define USE_UNALIGNED_MEMORY 1

//=============================================================================

#include <memory>
#include <cmath>
#include <complex>
#include <new>

#if __GNUC__ >= 3
#define force_inline  __attribute__((const,always_inline))
#define force_inline2 __attribute__((pure,always_inline))
#define force_inline3 __attribute__((always_inline))
#else
#define force_inline
#define force_inline2
#define force_inline3
#endif



//=============================================================================
// noexcept is not supported for all compilers
#if defined(__clang__)
#  if !__has_feature(cxx_noexcept)
#    define noexcept
#  endif
#else
#  if defined(__GXX_EXPERIMENTAL_CXX0X__) && __GNUC__ * 10 + __GNUC_MINOR__ >= 46 || \
      defined(_MSC_VER) && _MSC_VER >= 1900
	// everything fine ...
#  else
#    define noexcept
#  endif
#endif
//=============================================================================


#ifdef __MINGW32__
#include <malloc.h>
#endif

//=============================================================================
// User Defined Constants
//=============================================================================

namespace OTFFT
{
    namespace CONSTANT
    {
        extern const double SQRT2;
        extern const double PI;
        extern const double SQRT1_2;
        extern const double RSQRT2PSQRT2;
        extern const double H1X;
        extern const double H1Y;
    }

    enum scaling_mode { scale_1 = 0, scale_unitary, scale_length };
} // namespace OTFFT

//=============================================================================
// User Defined Complex Class
//=============================================================================

namespace OTFFT
{
    struct complex_t
    {
        double Re, Im;

        complex_t() noexcept : Re(0), Im(0) {}
        complex_t(const double& x) noexcept : Re(x), Im(0) {}
        complex_t(const double& x, const double& y) noexcept : Re(x), Im(y) {}
        complex_t(const complex_t& z) noexcept : Re(z.Re), Im(z.Im) {}
        complex_t(const std::complex<double>& z) : Re(z.real()), Im(z.imag()) {}
        operator std::complex<double>() { return std::complex<double>(Re, Im); }

        complex_t& operator+=(const complex_t& z) noexcept
        {
            Re += z.Re;
            Im += z.Im;
            return *this;
        }

        complex_t& operator-=(const complex_t& z) noexcept
        {
            Re -= z.Re;
            Im -= z.Im;
            return *this;
        }

        complex_t& operator*=(const double& x) noexcept
        {
            Re *= x;
            Im *= x;
            return *this;
        }

        complex_t& operator/=(const double& x) noexcept
        {
            Re /= x;
            Im /= x;
            return *this;
        }

        complex_t& operator*=(const complex_t& z) noexcept
        {
            const double tmp = Re*z.Re - Im*z.Im;
            Im = Re*z.Im + Im*z.Re;
            Re = tmp;
            return *this;
        }
    };

    typedef double* __restrict const double_vector;
    typedef const double* __restrict const const_double_vector;
    typedef complex_t* __restrict const complex_vector;
    typedef const complex_t* __restrict const const_complex_vector;

    static inline double Re(const complex_t& z) noexcept force_inline;
    static inline double Re(const complex_t& z) noexcept { return z.Re; }
    static inline double Im(const complex_t& z) noexcept force_inline;
    static inline double Im(const complex_t& z) noexcept { return z.Im; }

    static inline double norm(const complex_t& z) noexcept force_inline;
    static inline double norm(const complex_t& z) noexcept
    {
        return z.Re*z.Re + z.Im*z.Im;
    }
    static inline complex_t conj(const complex_t& z) noexcept force_inline;
    static inline complex_t conj(const complex_t& z) noexcept
    {
        return complex_t(z.Re, -z.Im);
    }
    static inline double abs(const complex_t& z) noexcept force_inline;
    static inline double abs(const complex_t& z) noexcept
    {
        return sqrt(norm(z));
    }
    static inline double arg(const complex_t& z) noexcept force_inline;
    static inline double arg(const complex_t& z) noexcept
    {
        return atan2(z.Im, z.Re);
    }
    static inline complex_t proj(const complex_t& z) noexcept force_inline;
    static inline complex_t proj(const complex_t& z) noexcept
    {
        const double den(norm(z) + double(1.0));
        return complex_t(double(2.0) * z.Re / den, double(2.0) * z.Im / den);
    }
    static inline complex_t polar(const double rho, const double theta) noexcept force_inline;
    static inline complex_t polar(const double rho, const double theta) noexcept
    {
        return complex_t(rho * cos(theta), rho * sin(theta));
    }
    static inline complex_t jx(const complex_t& z) noexcept force_inline;
    static inline complex_t jx(const complex_t& z) noexcept
    {
        return complex_t(-z.Im, z.Re);
    }
    static inline complex_t neg(const complex_t& z) noexcept force_inline;
    static inline complex_t neg(const complex_t& z) noexcept
    {
        return complex_t(-z.Im, -z.Re);
    }
    static inline complex_t v8x(const complex_t& z) noexcept force_inline;
    static inline complex_t v8x(const complex_t& z) noexcept
    {
        return complex_t(CONSTANT::SQRT1_2*(z.Re-z.Im), CONSTANT::SQRT1_2*(z.Re+z.Im));
    }
    static inline complex_t w8x(const complex_t& z) noexcept force_inline;
    static inline complex_t w8x(const complex_t& z) noexcept
    {
        return complex_t(CONSTANT::SQRT1_2*(z.Re+z.Im), CONSTANT::SQRT1_2*(z.Im-z.Re));
    }

    static inline complex_t operator+(const complex_t& a, const complex_t& b) noexcept force_inline;
    static inline complex_t operator+(const complex_t& a, const complex_t& b) noexcept
    {
        return complex_t(a.Re + b.Re, a.Im + b.Im);
    }
    static inline complex_t operator-(const complex_t& a, const complex_t& b) noexcept force_inline;
    static inline complex_t operator-(const complex_t& a, const complex_t& b) noexcept
    {
        return complex_t(a.Re - b.Re, a.Im - b.Im);
    }
    static inline complex_t operator*(const double& a, const complex_t& b) noexcept force_inline;
    static inline complex_t operator*(const double& a, const complex_t& b) noexcept
    {
        return complex_t(a*b.Re, a*b.Im);
    }
    static inline complex_t operator*(const complex_t& a, const complex_t& b) noexcept force_inline;
    static inline complex_t operator*(const complex_t& a, const complex_t& b) noexcept
    {
        return complex_t(a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re);
    }
    static inline complex_t operator/(const complex_t& a, const double& b) noexcept force_inline;
    static inline complex_t operator/(const complex_t& a, const double& b) noexcept
    {
        return complex_t(a.Re/b, a.Im/b);
    }
    static inline complex_t operator/(const complex_t& a, const complex_t& b) noexcept force_inline;
    static inline complex_t operator/(const complex_t& a, const complex_t& b) noexcept
    {
        const double b2 = b.Re*b.Re + b.Im*b.Im;
        return (a * conj(b)) / b2;
    }

    static inline complex_t expj(const double& theta) noexcept force_inline;
    static inline complex_t expj(const double& theta) noexcept
    {
        return complex_t(cos(theta), sin(theta));
    }
} // namespace OTFFT

//=============================================================================
// constexpr sqrt
//=============================================================================

namespace OTFFT
{
    static inline double sqrt_aux(double a, double x, double y)
    {
        return x == y ? x : sqrt_aux(a, (x + a/x)/2, x);
    }

    static inline double mysqrt(double x) { return sqrt_aux(x, x/2, x); }
} // namespace OTFFT
