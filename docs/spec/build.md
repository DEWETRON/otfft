Build and platforms
===================

Rules
-----

*   **BLD-1** `[done]` `cmake` configures the library and the tests on Linux.
    (CI: `.github/workflows/cmake.yml`, Ubuntu)
*   **BLD-2** `[built]` The library compiles with Visual Studio on Windows
    and GCC on Linux. CI (GitHub Actions) builds and tests both.
*   **BLD-3** `[built]` Backends `OTFFT_WITH_SSE2`, `OTFFT_WITH_AVX` and
    `OTFFT_WITH_AVX2` are switched in `CMakeLists.txt`. At least one must be
    on. Other targets get SSE2 only.
*   **BLD-4** `[built]` OpenMP is the standard one. Thread count follows
    `OMP_NUM_THREADS`.
*   **BLD-5** `[done]` `ctest` runs the doctest suites in `unit_tests/`. doctest
    is vendored in `third_party/doctest`, so no test dependency needs
    installing. `-DOTFFT_BUILD_TESTS=OFF` skips the tests.
*   **BLD-6** `[built]` Public headers: `otfft.h`, `otfft_complex.h`,
    `otfft_fwd.h`, `otfft_types.h`, plus the generated `otfft_config.h`.
*   **BLD-7** `[done]` `-DOTFFT_BUILD_TOOLS=ON` (default OFF) builds `ffttune`.
    For each power-of-two length from 2^1 to 2^24 it times the seven
    algorithms of one backend (`--backend sse2|avx|avx2`, default the widest
    built), checks each result against DIF4, and prints the fastest one per
    length and the matching `case` lines for the switch in
    `src/otfft_algo.cpp`. It changes no library file; applying the lines is a
    separate change. (`ffttune.smoke` in `ctest`; CI does not build the tools.)

Decided
-------

*   D3: one library per backend, chosen at run time.
*   D12: `ffttune` prints the algorithm table, it does not generate library files.

Questions
---------


*   None open.
