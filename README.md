# OTFFT
[![Github All Releases](https://img.shields.io/github/downloads/DEWETRON/otfft/total)]()

OTFFT is a high-speed FFT library using the Stockham algorithm, SIMD
(SSE2, AVX, AVX2) and C++ template metaprogramming. It is a mixed-radix FFT
with real-input FFT, DCT-II and Bluestein (any length) variants, and uses
OpenMP for large transforms.

OTFFT is developed by OK Ojisan (Takuya OKAHISA). The original homepage is
http://wwwa.pikara.ne.jp/okojisan/otfft-en/. This repository is the DEWETRON
fork.

# Build status
| CI | Status |
| ---:| ------ |
| GitHub Actions (Ubuntu, Windows) | [![CMake](https://github.com/DEWETRON/otfft/actions/workflows/cmake.yml/badge.svg)](https://github.com/DEWETRON/otfft/actions/workflows/cmake.yml) |

Built and tested with GCC on Linux and Visual Studio on Windows.

# The DEWETRON fork
The fork uses the original source code with these changes:
* abstract interfaces behind a factory (`OTFFT::Factory`), so the caller does not depend on one instruction set
* multi-CPU builds: SSE2, AVX and AVX2 backends, chosen at run time
* unit tests (doctest) and CI
* standard OpenMP
* benchmarks against FFTW, PocketFFT and KissFFT

## Status
Relevant changes from upstream version 11.5 are merged. The latest upstream
release is 11.6e (2023-12-21), which this fork does not follow yet. The
original classes (`OTFFT::FFT`, `RFFT`, `DCT`, `Bluestein`, `FFT0`, `DCT0`)
are internal here; use the factory functions below.

Planned work is in [docs/ROADMAP.md](docs/ROADMAP.md): newer AMD64 extensions,
ARM64, examples, and fixes for the problems found by the benchmarks
(issues [#16](https://github.com/DEWETRON/otfft/issues/16) to
[#19](https://github.com/DEWETRON/otfft/issues/19)).

# Build
You need CMake 3.22 or newer, a C++11 compiler and OpenMP.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
(cd build && ctest)          # unit tests
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `OTFFT_BUILD_TESTS` | ON | build the unit tests (doctest is included in `third_party/`) |
| `OTFFT_BUILD_BENCHMARKS` | OFF | build `otfft_bench`; finds FFTW with pkg-config if installed |
| `OTFFT_BUILD_TOOLS` | OFF | build `ffttune`, which finds the fastest algorithm for each power-of-two length on this machine |

To use the library from another CMake project, add the directory with
`add_subdirectory(otfft)` and link the `otfft` target. Include the header as
`#include "otfft.h"`.

# Using OTFFT
All transforms are created by `OTFFT::Factory` and returned as smart pointers.
The length is given on creation, or later with `setup(n)`. A transform object
holds an internal work buffer: **use one object per thread**, never one object
from several threads at once.

Buffers must be SIMD aligned: allocate them with `OTFFT::simd_malloc` and
release them with `OTFFT::simd_free`.

The backend is chosen automatically at run time. To force one, pass an
`OTFFT::OptimizationType` as second argument, for example
`createComplexFFT(n, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX)`.

## Complex-to-complex FFT
``` c++
#include "otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_free;
using OTFFT::simd_malloc;

void f(int N)
{
    auto fft = OTFFT::Factory::createComplexFFT(N);   // N is the sequence length
    complex_t* x = (complex_t*) simd_malloc(N * sizeof(complex_t));
    // fill x
    fft->fwd(x);   // x is input and output
    // use the spectrum
    fft->inv(x);   // back to the signal
    simd_free(x);
}
```

`complex_t` holds two doubles, `Re` and `Im`. The member functions are:

| Forward | Inverse | Scale |
| --- | --- | --- |
| `fwd(x)`, `fwdn(x)` | `invn(x)` | 1/N |
| `fwd0(x)` | `inv(x)`, `inv0(x)` | none |
| `fwdu(x)` | `invu(x)` | 1/sqrt(N) |

Note that `fwd` divides by N and `inv` does not, so `inv(fwd(x))` returns `x`.

To change the length: `fft->setup(2 * N);`

## Real-to-complex FFT
``` c++
auto rfft = OTFFT::Factory::createRealFFT(N);        // N must be even
double*    x = (double*)    simd_malloc(N * sizeof(double));
complex_t* y = (complex_t*) simd_malloc(N * sizeof(complex_t));
rfft->fwd(x, y);   // x is input, y is output; y gets the whole spectrum (N values)
rfft->inv(y, x);   // y is input and is destroyed, x is output
```

The same function names and scales as above apply.

## Discrete cosine transform (DCT-II)
The transform is not orthogonalised and works in place. N must be even.
``` c++
auto dct = OTFFT::Factory::createDCT(N);
dct->fwd(x);   // x is input and output (scaled by 1/N); fwd0 does not scale
dct->inv(x);
```
Available: `fwd`, `fwd0`, `fwdn` and `inv`, `inv0`, `invn`. `fwd0` returns
half of what FFTW's `REDFT10` and PocketFFT's DCT-II compute.

## Bluestein FFT (any length)
Bluestein's FFT works for any length, including large primes, in O(N log N).
``` c++
auto bst = OTFFT::Factory::createBluesteinFFT(N);
bst->fwd(x);   // same interface as the complex FFT
```
For prime lengths use this one: `createComplexFFT` is very slow for primes and
returns wrong results for some ([#16](https://github.com/DEWETRON/otfft/issues/16),
[#17](https://github.com/DEWETRON/otfft/issues/17)).

# Benchmarks
Complex FFT, one thread, equivalent MFLOPS (higher is better), AMD Ryzen 9
3900X, GCC 15.2:

| N | OTFFT | FFTW 3.3.10 | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 28,760 | 37,775 | 5,135 | 5,180 |
| 4,096 | 23,500 | 27,356 | 7,409 | 5,914 |
| 65,536 | 6,907 | 21,515 | 4,726 | 5,110 |
| 1,048,576 | 5,346 | 8,176 | 3,095 | 2,256 |

Same test on a Windows notebook, Intel Core Ultra 7 268V, MSVC (FFTW not
available there):

| N | OTFFT | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: |
| 256 | 42,646 | 5,950 | 8,536 |
| 4,096 | 25,566 | 9,123 | 9,406 |
| 65,536 | 13,639 | 4,717 | 6,346 |
| 1,048,576 | 6,280 | 5,012 | 2,801 |

OTFFT is the fastest of the three at every power of two on this machine; see
[docs/benchmarks.md](docs/benchmarks.md#intel-core-ultra-7-268v-windows-msvc).

* OTFFT is within 1.3x of FFTW up to N=16,384 and 1.3x to 26x faster than PocketFFT and KissFFT on powers of two. From N=65,536, FFTW is 1.6x to 3x faster.
* The DCT-II is 1.1x to 1.9x faster than FFTW's up to N=262,144.
* Lengths with factors 3, 5 and 7 are 2x to 11x slower than FFTW, and OTFFT's own prime-length path is much slower still.
* The upstream claim that OTFFT is faster than FFTW at every length is **not reproduced** on this machine; the section [Comparison with the upstream benchmark](docs/benchmarks.md#comparison-with-the-upstream-benchmark) shows the two side by side.

Method, caveats and all tables are in [docs/benchmarks.md](docs/benchmarks.md).
The numbers come from two machines, each shared with other work, and vary by up to 30% for large sizes.
To run them yourself: `cmake -B build -DOTFFT_BUILD_BENCHMARKS=ON`, then
`build/Release/otfft_bench` (see the benchmarks page).

# Documentation
* [docs/spec/](docs/spec/README.md): what the library does, rule by rule
* [docs/decisions.md](docs/decisions.md): why it is built this way
* [docs/benchmarks.md](docs/benchmarks.md): speed against FFTW, PocketFFT and KissFFT
* [docs/ROADMAP.md](docs/ROADMAP.md): what is planned
* [AGENTS.md](AGENTS.md): rules for people and tools working on the code

# License
OTFFT is released under the MIT license (see [LICENSE](LICENSE)). The benchmark
tool uses PocketFFT and KissFFT (BSD, in `third_party/`) and, if you install it,
FFTW (GPL; never linked into the library).
