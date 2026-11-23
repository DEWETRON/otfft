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

Decided
-------

*   D3: one library per backend, chosen at run time.

Questions
---------

