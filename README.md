# otfft 
OTFFT is a high-speed FFT library using the Stockham's
algorithm and AVX.  In addition, C++ template metaprogramming
technique is used in OTFFT. And OTFFT is a mixed-radix FFT.

# Build Status
[![travis-ci](https://travis-ci.org/DEWETRON/otfft.svg?branch=master)](https://travis-ci.org/DEWETRON/otfft)
[![Run Status](https://api.shippable.com/projects/5a5f2e8eb108ab0600ee3e3d/badge?branch=master)](https://app.shippable.com/github/DEWETRON/otfft) 
[![Build status](https://ci.appveyor.com/api/projects/status/1hxy8w9laeaf44ln?svg=true)](https://ci.appveyor.com/project/DEWETRON/otfft)

# Reasons for the fork
OTFFT is developed by OK Ojisan(Takuya OKAHISA). It's original
homepage is http://wwwa.pikara.ne.jp/okojisan/otfft-en/.

The DEWETRON fork uses the original source code with improvements:
* multi cpu ISS builds (AVX, SSE2,...)
* unit test coverage

At the time of this writing, there is no existing up-to-date fork.

# How to use
Just run cmake to get an appropriate build environment
depending on your used build system and operating system.


# Complex-to-Complex FFT
``` c++
    #include "otfft/otfft.h"
    using OTFFT::complex_t;
    using OTFFT::simd_malloc;
    using OTFFT::simd_free;

    void f(int N)
    {
        complex_t* x = (complex_t*) simd_malloc(N*sizeof(complex_t));
        // do something
        OTFFT::FFT fft(N); // creation of FFT object. N is sequence length.
        fft.fwd(x); // execution of transformation. x is input and output
        // do something
        simd_free(x);
}
```

complex_t is defined as follows.

``` c++
    struct complex_t
    {
        double Re, Im;

        complex_t() : Re(0), Im(0) {}
        complex_t(const double& x) : Re(x), Im(0) {}
        complex_t(const double& x, const double& y) : Re(x), Im(y) {}
        complex_t(const std::complex<double>& z) : Re(z.real()), Im(z.imag()) {}
        operator std::complex<double>(){ return std::complex(Re, Im); }

    // ...
    };
```

There are member functions, such as the following.

    fwd(x)  -- DFT(with 1/N normalization) x:input/output
    fwd0(x) -- DFT(non normalization) x:input/output
    fwdu(x) -- DFT(unitary transformation) x:input/output
    fwdn(x) -- DFT(with 1/N normalization) x:input/output

    inv(x)  -- IDFT(non normalization) x:input/output
    inv0(x) -- IDFT(non normalization) x:input/output
    invu(x) -- IDFT(unitary transformation) x:input/output
    invn(x) -- IDFT(with 1/N normalization) x:input/output

To change the FFT size, do the following.

    fft.setup(2 * N);

To use in a multi-threaded environment, we do as follows.

``` c++
    #include "otfft/otfft.h"
    using OTFFT::complex_t;
    using OTFFT::simd_malloc;
    using OTFFT::simd_free;

    void f(int N)
    {
        complex_t* x = (complex_t*) simd_malloc(N*sizeof(complex_t));
        complex_t* y = (complex_t*) simd_malloc(N*sizeof(complex_t));
        // do someting
        OTFFT::FFT0 fft(N);
        fft.fwd(x, y); // x is input/output. y is work area
        // do something
        fft.inv(x, y);
        // x is input/output. y is work area
        // do someting
        simd_free(y);
        simd_free(x);
    }
```

Please note that "OTFFT::FFT" was changed to "OTFFT::FFT0".

# Real-to-Complex FFT
``` c++
    #include "otfft/otfft.h"
    using OTFFT::complex_t;
    using OTFFT::simd_malloc;
    using OTFFT::simd_free;

    void f(int N)
    {
        double* x = (double*) simd_malloc(N*sizeof(double));
        complex_t* y = (complex_t*) simd_malloc(N*sizeof(complex_t));
        // do something
        OTFFT::RFFT rfft(N);
        rfft.fwd(x, y); // x is input. y is output
        // do something
        simd_free(y);
        simd_free(x);
        }
```

N must be an even number. There are member functions, such as the
following.

    fwd(x, y)  -- DFT(with 1/N normalization) x:input, y:output
    fwd0(x,y)  -- DFT(non normalization) x:input, y:output
    fwdu(x, y) -- DFT(unitary transformation) x:input, y:output
    fwdn(x, y) -- DFT(with 1/N normalization) x:input, y:output

    inv(y, x)  -- IDFT(non normalization) y:input, x:output
    inv0(y, x) -- IDFT(non normalization) y:input, x:output
    invu(y, x) -- IDFT(unitary transformation) y:input, x:output
    invn(y, x) -- IDFT(with 1/N normalization) y:input, x:output

inv,inv0,invu,invn will destroy the input.


# Discrete Cosine Transformation(DCT-II)

This transformation, orthogonalization is not executed.
``` c++
    #include "otfft/otfft.h"
    using OTFFT::complex_t;
    using OTFFT::simd_malloc;
    using OTFFT::simd_free;

    void f(int N)
    {
        double* x = (double*) simd_malloc(N*sizeof(double));
        // do something
        OTFFT::DCT dct(N);
        dct.fwd(x); // execution of DCT. x is input and output
        // do something
        simd_free(x);
    }
```

N must be an even number. There are member functions, such as the
following.

    fwd(x)  -- DCT(with 1/N normalization) x:input/output
    fwd0(x) -- DCT(non normalization) x:input/output
    fwdn(x) -- DCT(with 1/N normalization) x:input/output

    inv(x) -- IDCT(non normalization) x:input/output
    inv0(x) -- IDCT(non normalization) x:input/output
    invn(x) -- IDCT(with 1/N normalization) x:input/output

To use in a multi-threaded environment, we do as follows.

``` c++
    #include "otfft/otfft.h"
    using OTFFT::complex_t;
    using OTFFT::simd_malloc;
    using OTFFT::simd_free;

    void f(int N)
    {
        double* x = (double*) simd_malloc(N*sizeof(double));
        double* y = (double*) simd_malloc(N*sizeof(double));
        complex_t* z = (complex_t*) simd_malloc(N*sizeof(complex_t));
        // do something
        OTFFT::DCT0 dct(N);
        dct.fwd(x, y, z); // x is input/output. y,z are work area
        // do something
        dct.inv(x, y, z); // x is input/output. y,z are work area
        // do somthing
        simd_free(z);
        simd_free(y);
        simd_free(x);
    }
```

Please note that "OTFFT::DCT" was changed to "OTFFT::DCT0".

# Bluestein's FFT

Bluestein's FFT is the FFT of any sequence length. Even if the
sequence length is a big prime number, the order of complexity is O(N
log N).
``` c++
    #include "otfft/otfft.h"
    using OTFFT::complex_t;
    using OTFFT::simd_malloc;
    using OTFFT::simd_free;

    void f(int N)
    {
        complex_t* x = (complex_t*) simd_malloc(N*sizeof(complex_t));
        // do something
        OTFFT::Bluestein bst(N);
        bst.fwd(x); // execution of Bluestein's FFT. x is input and output
        // do something
        simd_free(x);
    }
```

There are member functions, such as the following.

    fwd(x)  -- DFT(with 1/N normalization) x:input/output
    fwd0(x) -- DFT(non normalization) x:input/output
    fwdu(x) -- DFT(unitary transformation) x:input/output
    fwdn(x) -- DFT(with 1/N normalization) x:input/output

    inv(x)  -- IDFT(non normalization) x:input/output
    inv0(x) -- IDFT(non normalization) x:input/output
    invu(x) -- IDFT(unitary transformation) x:input/output
    invn(x) -- IDFT(with 1/N normalization) x:input/output

To use in a multi-threaded environment, we need to create objects of
the same number as the number of threads.
