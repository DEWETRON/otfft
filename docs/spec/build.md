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
    on. Other targets get SSE2 only. `OTFFT_WITH_AVX512` is an option, see BLD-10.
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
    `src/otfft_algo.cpp`. It changes no library file. With `--out FILE` it writes
    a tuning file (format: FAC-10): the `serial` table for one thread
    (`--threads 1`, the default) or the `threaded` table for more, with the CPU model,
    compiler and run settings as comment lines. It reads the file back with
    `OTFFT::loadTuningFile` and fails if the library rejects it. The
    built-in tables are changed only in a separate change.
    (`ffttune.smoke`, `ffttune.tuning_file` and `ffttune.tuning_file_threaded` in
    `ctest`; CI does not build the tools.)

*   **BLD-8** `[done]` `-DOTFFT_BUILD_EXAMPLES=ON` builds the programs in
    `examples/` (`complex_fft`, `real_fft`, `dct`, `bluestein`). The default is
    ON when otfft is the top-level project and OFF when it is added with
    `add_subdirectory`. Each program prints its results, compares them with a
    direct formula or a round trip, and returns 0 only if all checks pass.
    `ctest` runs them as `example.<name>` and also requires the line
    `<name>: ok`, so CI builds and runs them on Linux and Windows.

*   **BLD-9** `[done]` The three backends are compiled for the instruction set they are named
    for: SSE2 without AVX, AVX with AVX and without AVX2 and FMA, AVX2 with AVX2 and FMA. GCC
    and Clang get `-msse2`, `-mavx` and `-mavx2 -mfma`; MSVC `/arch:AVX` and `/arch:AVX2` (which
    includes FMA). (`otfft_backend_test: TestSse2BackendUsesNoAvx,
    TestAvxBackendUsesAvxWithoutFma, TestAvx2BackendUsesAvx2AndFma`; the Intel compiler branch
    in `CMakeLists.txt` is not covered.)

*   **BLD-10** `[done]` `-DOTFFT_WITH_AVX512=ON` (default OFF, 64 bit only) builds a fourth backend,
    `otfft_avx512`, from the same sources: GCC and Clang get `-mavx2 -mfma -mavx512f -mavx512dq
    -mavx512vl`, MSVC `/arch:AVX512`; the Intel compiler stops with an error. The other backends
    are not compiled with AVX-512. (D32.) (`otfft_backend_test: TestAvx512BackendUsesAvx512WithAvx2AndFma`
    and the `compiledWithAvx512` checks of the other backends; the GCC branch was built and tested, the
    Clang and MSVC branches were **not built**.)
*   **BLD-11** `[done]` With clang-cl (`-T ClangCL` in the Visual Studio generator) the library builds with the MSVC
    options: `/arch:` for the backends, `-mfma` added for AVX2 and AVX-512 (clang-cl does not enable FMA with
    `/arch:AVX2`), `-msse3` added for SSE2 (the SSE2 code uses SSE3 intrinsics). OpenMP needs `libomp` from
    the Visual Studio LLVM given to CMake by hand, see `.github/workflows/cmake.yml`. (D34.)
    (`otfft_backend_test`, and the whole suite in the CI job "windows-latest, clang-cl")

Decided
-------

*   D3: one library per backend, chosen at run time.
*   D12: `ffttune` prints the algorithm table, it does not generate library files.
*   D23: `ffttune --out` writes a tuning file, not `case` lines.
*   D18: the examples check themselves and run under `ctest`.
*   D21: the AVX2 backend is kept and built with FMA.

Questions
---------


*   None open.
