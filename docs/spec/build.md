Build and platforms
===================

Rules
-----

*   **BLD-1** `[done]` `cmake` configures the library and the tests on Linux.
    (CI: `.github/workflows/cmake.yml`, Ubuntu)
*   **BLD-2** `[built]` The library compiles with Visual Studio on Windows
    (AppVeyor, VS2013 and VS2017) and GCC on Linux.
*   **BLD-3** `[built]` Backends `OTFFT_WITH_SSE2`, `OTFFT_WITH_AVX` and
    `OTFFT_WITH_AVX2` are switched in `CMakeLists.txt`. At least one must be
    on. Other targets get SSE2 only.
*   **BLD-4** `[built]` OpenMP is the standard one. Thread count follows
    `OMP_NUM_THREADS`.
*   **BLD-5** `[done]` `ctest` runs the Boost.Test suites in `unit_tests/`.
    Boost.Test is required.
*   **BLD-6** `[built]` Public headers: `otfft.h`, `otfft_complex.h`,
    `otfft_fwd.h`, `otfft_types.h`, plus the generated `otfft_config.h`.

Decided
-------

*   D3: one library per backend, chosen at run time.

Questions
---------

*   `.travis.yml`, `.shippable.yml` and `appyeyor.yml` (sic) look stale. Remove?
