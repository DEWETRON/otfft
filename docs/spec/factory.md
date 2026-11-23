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

Seen in upstream
----------------

*   Upstream has concrete classes (`FFT`, `RFFT`, `DCT`); the abstract
    interfaces and the factory are an addition of this fork.

Decided
-------

*   D2: factory with abstract interfaces. D3: one library per CPU backend.

Questions
---------

*   Should `setup(n)` with an unsupported `n` (negative, above 2^30) report an
    error? Today the methods are `noexcept` and nothing is reported.
