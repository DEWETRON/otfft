Factory
=======

Transforms are created through `OTFFT::Factory`. The caller owns the result.

Rules
-----

*   **FAC-1** `[done]` `createComplexFFT`, `createRealFFT`, `createDCT` and
    `createBluesteinFFT` each return a smart pointer to a ready object.
    (`otfft_basic_test: TestFactory`)
*   **FAC-2** `[built]` The length argument `n` defaults to 0. An object made
    with 0 must be given a length with `setup(n)` before use.
*   **FAC-3** `[built]` `setup(n)` may be called again to change the length.
    Lengths up to 2^30 are supported.
*   **FAC-4** `[done]` `OPTIMIZED_FFT_AUTO` picks the best backend the CPU
    supports at run time. (`otfft_optimization_test`)
*   **FAC-5** `[done]` A caller may force `SSE2`, `AVX` or `AVX2`. The values
    exist only if the library was built with that backend.
    (`otfft_optimization_test`)
*   **FAC-6** `[built]` All transform methods are `const noexcept`, but a
    complex, real or DCT object owns an internal work buffer (`FFT::work` in
    `src/otfft_base.h`) that every call uses. **Do not share one object between
    threads;** create one object per thread, as `otfft_basic_test: TestFactory`
    does.
*   **FAC-7** `[built]` Buffers passed to transforms must be SIMD-aligned.
    Use `OTFFT::simd_malloc` and `simd_free`.
*   **FAC-8** `[built]` The AVX2 backend is chosen (automatically or forced) only on a CPU
    that has AVX2 **and** FMA, because it is compiled with FMA (D21); otherwise the AVX
    backend is used. No test fakes the CPU, so only the build side is tested (BLD-9).
*   **FAC-9** `[done]` `OTFFT::loadTuningFile(path, error)` replaces the built-in
    per-length algorithm choice of the complex FFT for the lengths a text file lists.
    Lengths it does not list keep the built-in choice. Without a call nothing changes.
    (`otfft_tuning_test: TestLoadSetsListedEntriesOnly`, `TestNothingLoadedByDefault`)
*   **FAC-10** `[done]` File format, one entry per line, `#` starts a comment, empty lines
    are ignored. The first line is the header `otfft-tuning 1`. An entry is
    `<backend> <table> <log2 N> <algorithm>`: backend `sse2`, `avx` or `avx2`; table
    `serial` (one OpenMP thread) or `threaded` (more than one, used from length 2^15,
    see CFFT-8); log2 N from 1 to 24; algorithm `DIF4`, `DIT4`, `DIF8`, `DIT8`, `DIF16`,
    `DIT16` or `SixStep`. If a backend, table and length are listed twice the last entry
    wins. A backend that the library was not built with is accepted and has no effect.
    (`TestAllAlgorithmNames`, `TestLastEntryWins`)
*   **FAC-11** `[done]` A file with any error (cannot be opened, no or wrong header, a
    malformed entry) is rejected as a whole: the call returns `false`, `error` names the line
    and the problem, and the tuning loaded before stays. (`TestBadFilesAreRejectedAndChangeNothing`,
    `TestErrorNamesTheLine`, `TestMissingFile`)
*   **FAC-12** `[done]` A successful call replaces the tuning loaded before, it does not add
    to it. `OTFFT::resetTuning()` drops it: the built-in choice applies again.
    (`TestLoadReplacesEarlierFile`, `TestResetDropsTheFile`)
*   **FAC-13** `[done]` The table is read when a transform is set up (`createComplexFFT(n)` or
    `setup(n)`). An object keeps its algorithm if a file is loaded later. Every algorithm
    gives the right spectrum when chosen by a file, with one thread and with several.
    The file is not checked against the machine: a table made on another CPU is used as
    it is. Loading is safe while other threads set up transforms (they see the old or the
    new table). (`TestEveryAlgorithmGivesTheRightSpectrumOneThread`,
    `TestEveryAlgorithmGivesTheRightSpectrumSeveralThreads`,
    `TestPlanKeepsItsAlgorithmWhenTheFileChanges`)

Seen in upstream
----------------

*   Upstream has concrete classes (`FFT`, `RFFT`, `DCT`); the abstract
    interfaces and the factory are an addition of this fork.

Decided
-------

*   D2: factory with abstract interfaces. D3: one library per CPU backend.
*   D22: run-time tuning file for the algorithm tables.

Questions
---------

*   Should `setup(n)` with an unsupported `n` (negative, above 2^30) report an
    error? Today the methods are `noexcept` and nothing is reported.
