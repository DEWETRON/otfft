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
*   [ ] **R1.3** Add examples: small buildable programs for the complex,
    real, DCT and Bluestein transforms, built by CMake.

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
*   [ ] **R1.6** OpenMP: do not thread below the break-even size (N=16,384 is
    2.5x slower with 12 threads) and scale better above it (1.8x at 4M points,
    [#19](https://github.com/DEWETRON/otfft/issues/19)).
*   [ ] **R1.7** Decide whether the AVX2 backend is worth keeping (no gain over
    AVX in the benchmarks); recheck after R2.1.

Milestone 1c: Catch up with upstream 11.6e
------------------------------------------

Background and measurements: `docs/upstream-comparison.md`.

*   [ ] **R1.8** Re-tune the per-size algorithm table
    (`src/otfft_algo.cpp`) for the machines we care about: port upstream's
    `ffttune` or add a run-time tuner. Upstream's table alone gave +28% to
    +94% for N = 2^16 to 2^20 on the benchmark machine.
    Done: the tool, `tools/ffttune.cpp` (`-DOTFFT_BUILD_TOOLS=ON`, D12). Still
    open: run it on the machines we care about (a baseline for one Intel
    notebook is in `docs/benchmarks.md`) and apply the table.
    Measured target (2026-11-24, `docs/benchmarks.md`): the fork reaches 0.5 to
    0.6 of upstream 11.6e for complex N = 65,536 to 524,288 with one thread.
*   [ ] **R1.9** Port the `exp` kernels (`otfft_avx{dif,dit}{8,16}exp.h`,
    `init_Wq`, `init_Wr`, `EXP_THRESHOLD = 2^18`) from upstream 11.6e.
*   [ ] **R1.10** Port the OpenMP changes of 11.6e: two six-step thresholds,
    `schedule(static)` and `guided`, `omp sections` in the eight-step; then
    re-measure for issue [#19](https://github.com/DEWETRON/otfft/issues/19).
    Measured target: real and DCT with 12 threads from N = 262,144 are 0.5 to
    0.8 of upstream.
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
