Roadmap
=======

What we plan to do next. Items have a stable id (`R<milestone>.<n>`). Mark
finished items with `[x]` and keep them; do not renumber. Behaviour changes
also go in `docs/spec/`, design choices in `docs/decisions.md`.

Milestone 1: Foundations
------------------------

*   [x] **R1.1** Run benchmarks: measure the transforms (complex, real, DCT,
    Bluestein) per backend and length, and record results in
    `docs/benchmarks.md`. Compared with FFTW, PocketFFT and KissFFT.
*   [x] **R1.2** Remove dependencies: Boost.Test replaced by vendored doctest
    (D8). The library itself never depended on Boost.
*   [x] **R1.3** Add examples: small buildable programs for the complex,
    real, DCT and Bluestein transforms, built by CMake.
    Done: `examples/` (D18, BLD-8). Each program checks its own result and
    `ctest` runs it. Not covered: the C++ API of a project that uses the
    library through `add_subdirectory` (no example project yet).

Milestone 1b: Fixes and speed found by the benchmarks
-----------------------------------------------------

*   [x] **R1.4** Prime lengths with `createComplexFFT`: wrong result from a
    prime factor of 46,341 (int overflow, [#16](https://github.com/DEWETRON/otfft/issues/16))
    and O(N^2) speed ([#17](https://github.com/DEWETRON/otfft/issues/17)). Fix, or route to
    Bluestein, and add a unit test for prime lengths.
    Done: lengths with a prime factor above 100 use Bluestein (D13,
    `unit_tests/otfft_prime_test.cpp`). Still open: the real FFT and DCT with
    such a half length (spec RFFT-6).
*   [ ] **R1.5** Lengths with factors 3, 5 and 7 are 2x to 11x slower than FFTW
    (worst: N=10,080). Add a radix-7 butterfly and improve the mixed radix
    steps ([#18](https://github.com/DEWETRON/otfft/issues/18)).
    Done: the radix-7 butterfly (D15; a factor 7 is 1.9x to 5.5x faster) and
    the symmetric radix 3 and 5 butterflies (D16; a factor 5 is 1.1x to 1.4x
    faster, powers of 3 are unchanged). Measured against FFTW 3.3.10 (GCC, WSL2,
    Intel notebook): lengths with 3, 5 and 7 are now 1.9x to 2.2x slower than
    FFTW (N = 480, 1,000, 3,000, 10,080, 100,000; before 2.8x to 3.2x), so the
    "2x to 11x" of this item is now about 2x. Still open: close the rest of the
    gap (pass structure, radix 9, AVX versions of the passes; not analysed yet)
    and repeat the FFTW comparison on the Ryzen.
*   [ ] **R1.6** OpenMP: do not thread below the break-even size (N=16,384 is
    2.5x slower with 12 threads) and scale better above it (1.8x at 4M points,
    [#19](https://github.com/DEWETRON/otfft/issues/19)).
    Done: the first part (D17): per-class thresholds at the measured break-even and
    the serial table below 2^15 with several threads (N=16,384 with 4 and 8
    threads on the Intel notebook: from a collapse to the 1-thread speed). Still
    open: thresholds of the eight-step, mixed radix, real FFT, DCT and
    Bluestein code, and a check on the Ryzen with the changed library. Scaling
    above the break-even improved for the lengths where the table uses SixStep
    (R1.10: SixStep 1.5x to 2.5x faster from 2^19, N = 4,194,304 with 8 threads
    1.19x complex, 1.82x real); the table still picks SixStep where DIF8 is
    faster on the Intel notebook (R1.8).
*   [x] **R1.7** Decide whether the AVX2 backend is worth keeping (no gain over
    AVX in the benchmarks); recheck after R2.1.
    Decided: keep it (D21). The "no gain" was a build flag gap: GCC and Clang
    do not enable FMA with `-mavx2` and the flags had no `-mfma`. With it the
    AVX2 backend is 5% to 7% faster over all sizes and 10% to 18% faster at N = 64 to
    4,096 (GCC 15.2, one thread, Intel notebook); MSVC already had it. The
    backend is now chosen only on CPUs with AVX2 and FMA, and
    `otfft_backend_test` checks the flags. Not done: a run on the Ryzen and with
    Clang; the recheck after R2.1 (AVX-512) stays.

Milestone 1c: Catch up with upstream 11.6e
------------------------------------------

Background and measurements: `docs/upstream-comparison.md`.

*   [ ] **R1.8** Re-tune the per-size algorithm table
    (`src/otfft_algo.cpp`) for the machines we care about: port upstream's
    `ffttune` or add a run-time tuner. Upstream's table alone gave +28% to
    +94% for N = 2^16 to 2^20 on the benchmark machine.
    Done: the tool, `tools/ffttune.cpp` (`-DOTFFT_BUILD_TOOLS=ON`, D12), the
    baselines on an Intel and an AMD CPU, and a new table for one thread
    (D14, 1.15x to 1.87x faster from N = 16,384 on the Intel notebook). Still
    open: re-tune the table used with more than one thread (it is the old
    table; one 12-thread run exists, with odd results at 2^19 and 2^21), and
    check the new serial table on the Ryzen with the library, not only with
    `ffttune`.
    After R1.9 the best class at 2^18 to 2^22 differs between MSVC and GCC
    (`docs/benchmarks.md`, "Exp kernels (R1.9)"), so the table has to be tuned
    again for the exp kernels, on more than one machine and compiler.
    Measured target (2026-11-24, `docs/benchmarks.md`): the fork reaches 0.5 to
    0.6 of upstream 11.6e for complex N = 65,536 to 524,288 with one thread.
*   [x] **R1.9** Port the `exp` kernels (`otfft_avx{dif,dit}{8,16}exp.h`,
    `init_Wq`, `init_Wr`, `EXP_THRESHOLD = 2^18`) from upstream 11.6e.
    Done: the four kernel files, `EXP_THRESHOLD = 2^18` and a test of every
    class, backend and function (D19, CFFT-12). `init_Wq` and `init_Wr` are not
    used by upstream 11.6e, so they were not ported. Not a speed gain yet: on the
    Intel notebook the best class changes by 0.93x to 1.13x (DIT16 is faster on both
    compilers, DIF16 only with GCC); the Ryzen, where the gap to FFTW is largest,
    has not been run. The table is not retuned for the new kernels (see R1.8).
*   [x] **R1.10** Port the OpenMP changes of 11.6e: two six-step thresholds,
    `schedule(static)` and `guided`, `omp sections` in the eight-step; then
    re-measure for issue [#19](https://github.com/DEWETRON/otfft/issues/19).
    Measured target: real and DCT with 12 threads from N = 262,144 are 0.5 to
    0.8 of upstream.
    Done (D20, CFFT-13): the three changes and a test of every variant. The
    eight-step code from 2^19 ran its six-step parts on one thread (a missing
    `omp parallel`, not in upstream); now SixStep is 1.5x to 2.5x faster there
    (4 and 8 threads, Intel notebook), and through the library with 8 threads
    N = 4,194,304 is 1.19x (complex), 1.82x (real) and 1.58x (DCT) faster. The
    guided threshold is 2^17, not upstream's 2^16 (0.7x to 0.8x at N = 2^16).
    Not done: the comparison with upstream at 12 threads and a run on the Ryzen;
    at N = 262,144 nothing changed measurably.
*   [ ] **R1.11** After R1.8 to R1.10: update the version strings (still
    11.4xv) and the sync statement in the README. Try to get upstream 11.5e
    (the download link returned 404) to find which changes came when.
*   [ ] **R1.12** Report the `(i*j)%k` overflow and the O(N^2) generic radix
    (issues #16 to #18) to the upstream author; the code is identical in
    11.6e.
*   [ ] **R1.13** Decide whether to take over the C99 and Fortran 2003
    interfaces, and the upstream benchmark programs, from 11.6e.

Milestone 2: Platforms
----------------------

*   [ ] **R2.1** Support newer AMD64 instruction set extensions (beyond
    SSE2, AVX and AVX2, for example AVX-512), selected at run time like the
    existing backends (D3).
*   [ ] **R2.2** Support ARM64 (NEON), including a CI job and the same
    run-time backend choice.

Notes
-----

*   R1.1 comes first: it gives the baseline R2.1 and R2.2 are judged against.
