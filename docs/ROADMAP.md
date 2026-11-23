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

*   [ ] **R1.4** Prime lengths with `createComplexFFT`: wrong result at
    N=100,003 and O(N^2) speed at 1,009 and 10,007. Fix, or route to Bluestein,
    and add a unit test for prime lengths.
*   [ ] **R1.5** Lengths with factors 3, 5 and 7 are 2x to 11x slower than FFTW
    (worst: N=10,080). Improve the mixed radix steps.
*   [ ] **R1.6** OpenMP: do not thread below the break-even size (N=16,384 is
    2.5x slower with 12 threads) and scale better above it (1.8x at 4M points).
*   [ ] **R1.7** Decide whether the AVX2 backend is worth keeping (no gain over
    AVX in the benchmarks); recheck after R2.1.

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
