Comparison with upstream OTFFT 11.6e
====================================

How the source of this fork differs from the latest upstream release, and what
that means for speed. Done on 2026-11-23 by diffing this repository's `src/`
and `inc/` against `otfft-11.6e.tar.gz` (dated 2023-12-21 on
<http://wwwa.pikara.ne.jp/okojisan/otfft-en/download.html>).

Limits of this comparison:

*   The 11.5e tarball (link on the same page) returned 404, so it is not
    known which differences come from 11.5 and which from 11.6.
*   Differences were found by diffing after removing whitespace, namespace
    wrappers and version strings; they were read, not tested, except where a
    measurement is given.
*   Upstream is a C++ library with its own build files (Makefiles); only the
    sources were compared.

Summary
-------

*   The **algorithm kernels** (AVX radix 4/8/16 DIF/DIT, mixed radix) are the
    same as 11.6e. The mixed radix file is identical apart from the version
    string, a namespace wrapper and the `FFT0` object moved to
    `src/otfft_algo_mixed.cpp`. So the bugs in issues
    [#16](https://github.com/DEWETRON/otfft/issues/16) to
    [#18](https://github.com/DEWETRON/otfft/issues/18) (int overflow,
    O(N^2) generic radix, no radix 7) exist upstream too.
*   The fork is **behind** in three places: the `exp` kernels for large N are
    missing, the per-size algorithm table differs in 15 of 24 entries, and
    the OpenMP thresholds and scheduling differ.
*   The fork's source files still say version **11.4xv**.

What the fork added
-------------------

*   Abstract interfaces (`ComplexFFT`, `RealFFT`, `RealDCT`) and
    `OTFFT::Factory` (`inc/otfft.h`, `src/otfft_factory.h`, `src/otfft_if.h`).
*   Run-time choice of an SSE2, AVX or AVX2 backend (`src/otfft_sse2.h`,
    `otfft_avx.h`, `otfft_avx2.h`, `otfft_platform.h`). Upstream is compiled
    for one instruction set.
*   The `FFT0` objects moved from the headers into `src/otfft_algo_*.cpp`, and
    the algorithm is chosen in `src/otfft_algo.cpp` at run time instead of
    through generated `otfft_gen_*.h` files at compile time.
*   `abs`, `arg`, `proj` and `polar` for `complex_t` (`inc/otfft_complex.h`),
    and a CMake build, unit tests and benchmarks.

What upstream has that the fork does not
-----------------------------------------

*   **`exp` kernels**: `otfft_avxdif8exp.h`, `otfft_avxdif16exp.h`,
    `otfft_avxdit8exp.h`, `otfft_avxdit16exp.h`, used when N >= 2^18
    (`EXP_THRESHOLD = 1<<18`) together with the weight table builders
    `init_Wq` and `init_Wr` in `otfft_misc.h`. These are the large-N paths,
    where the benchmarks show the fork 1.6 to 3 times behind FFTW
    (see `docs/benchmarks.md`).
*   **`ffttune`** (`otfft/ffttune.cpp`): measures which algorithm is fastest
    for each length on the machine and generates the selection tables.
*   The **C99 and Fortran 2003** interfaces (`otfft_c.h`, `otfft.f03`).
*   The FFTW, Ooura and simple-FFT benchmark programs (`fftbench*.cpp`).

Algorithm table per size
------------------------

Both use one algorithm per power-of-two length. Upstream's table is generated
by `ffttune` on the author's machine; the fork's is fixed in
`src/otfft_algo.cpp`.

| log2 N | Upstream 11.6e | This fork |
| ---: | --- | --- |
| 1 | DIF4 | DIT4 |
| 2 | SixStep | DIF4 |
| 3 | DIT8 | DIT8 |
| 4 | DIT4 | DIF4 |
| 5 | DIT8 | DIT8 |
| 6 | DIT8 | DIF8 |
| 7 | DIF16 | DIF16 |
| 8 | DIF16 | DIF8 |
| 9 | DIF8 | DIF8 |
| 10 | DIF4 | DIF4 |
| 11 | DIF16 | DIF16 |
| 12 | DIF8 | DIT8 |
| 13 | DIF8 | DIF8 |
| 14 | DIT8 | DIT8 |
| 15 | DIF8 | DIT8 |
| 16 | DIF16 | SixStep |
| 17 | DIF16 | DIT8 |
| 18 | DIF16 | SixStep |
| 19 | DIF16 | SixStep |
| 20 | DIF16 | SixStep |
| 21 | SixStep | SixStep |
| 22 | DIF16 | SixStep |
| 23 | DIF16 | SixStep |
| 24 | DIF16 | SixStep |

Different entries: 15 of 24. Upstream's table is tuned for another machine, so
it is not automatically better here. A test shows it matters, however:

**Test:** upstream's table was put into `src/otfft_algo.cpp` in a throwaway
copy of `master`, and `otfft_bench --kinds complex --sizes pow2 --threads 1
--lib OTFFT --samples 7` was run for both builds (AMD Ryzen 9 3900X, GCC 15.2,
one shared machine, one run each). Equivalent MFLOPS, higher is better:

| N | Fork table | Upstream table | Change | FFTW (estimate) |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 21,470 | 21,163 | -1% | 20,846 |
| 64 | 28,093 | 26,461 | -6% | 30,741 |
| 256 | 28,846 | 31,404 | +9% | 37,775 |
| 1,024 | 27,953 | 29,515 | +6% | 35,771 |
| 4,096 | 23,769 | 19,972 | -16% | 27,356 |
| 16,384 | 17,996 | 18,776 | +4% | 19,744 |
| 65,536 | 7,227 | 14,018 | **+94%** | 21,515 |
| 262,144 | 8,347 | 14,943 | **+79%** | 16,624 |
| 1,048,576 | 5,341 | 6,851 | **+28%** | 8,176 |
| 4,194,304 | 5,291 | 5,564 | +5% | 4,093 |

Up to 2^15 the changes are within the noise of the benchmark except possibly
N=4,096. From 2^16 up, upstream's choice (DIF16 instead of SixStep) is much
faster here. The table is a good first target for re-tuning on the machines we
care about.

OpenMP differences
------------------

Thresholds are the sizes from which the parallel code is used.

| File | Upstream 11.6e | This fork |
| --- | --- | --- |
| `otfft_avxdit4.h` | 2^11 | 2^13 |
| `otfft_avxdit8.h` | 2^12 | 2^13 |
| `otfft_avxdit16.h` | 2^12 | 2^11 |
| `otfft_avxdif8.h` | 2^13 | 2^11 |
| `otfft_avxdif16.h` | 2^13 | 2^12 |
| `otfft_avxdif4.h` | 2^13 | 2^13 (same) |
| `otfft_sixstepsq.h` | two thresholds: 2^14 (part 1) and 2^16 (part 2) | one threshold, 2^14 |

Other differences in the parallel code:

*   `otfft_sixstepsq.h`: upstream uses `schedule(static)`, and `guided` in
    `part3`, plus a middle branch (only part 2 parallel) between the two
    thresholds. The fork uses the default schedule.
*   `otfft_eightstep.h`: upstream runs the eight sub-transforms in
    `omp sections`; the fork uses `omp parallel firstprivate(...)`.
*   The fork added `firstprivate(x,y,W)` to the `omp parallel` lines of the
    AVX OpenMP headers.

This is a likely contributor to
[#19](https://github.com/DEWETRON/otfft/issues/19) (N=16,384 slower with 12
threads and poor scaling). It was not tested.

Smaller differences
-------------------

*   `complex_t` constructors take `const double&` upstream and `double` here;
    upstream defines the copy constructor and assignment by hand.
*   Upstream's `otfft_misc.h` has the `exp` weight builders; the fork's is
    about 80 lines shorter and includes `otfft_types.h` where upstream
    includes `otfft_complex.h`.
*   `AVX_THRESHOLD = 1<<10` in the fork's `otfft_avxdif4.h`; upstream has no
    such constant. The commented-out `OMP_THRESHOLD = 1<<15` lines in the fork's
    AVX headers are leftovers.

Consequences
------------

*   The README's statement that changes from 11.5 are merged cannot be
    confirmed from the source: the `exp` kernels and the six-step changes are
    missing, and every header still says 11.4xv.
*   The speed gap to FFTW at large N is partly a tuning and porting gap, not
    a limit of the algorithms.
*   The prime and factor-7 problems are in upstream's code too and should also
    be reported to the author.

Follow-up work is in `docs/ROADMAP.md` (R1.8 to R1.12).
