DCT
===

`RealDCT`: in-place DCT-II on `double_vector`, and its inverse.

Rules
-----

*   **DCT-1** `[done]` `fwd*` computes the DCT-II. The transform is not
    orthogonalised. (`otfft_dct_test: TestConstantInput, TestAlternatingOnes`)
*   **DCT-2** `[done]` Cosine tables are cached and reused between calls.
    (`TestCachedCosines`)
*   **DCT-3** `[built]` N must be even.
*   **DCT-4** `[built]` Normalisation: `fwd`, `fwdn` scale by 1/N; `fwd0`,
    `inv`, `inv0` do not; `invn` scales by 1/N. There is no unitary variant.
*   **DCT-5** `[done]` Matches a reference implementation on the optimisation
    test data. (`otfft_optimization_test: TestDCT`)
*   **DCT-6** `[done]` The loops around the real FFT run on one thread below N = 2^20 whatever the
    thread count, and on several threads from there on. The result does not depend on the thread
    count, and `invn` after `fwd0` gives back the input. (D30.)
    (`otfft_omp_threshold_test: TestDctAroundThreshold`)

Seen in upstream
----------------

*   `DCT` and `DCT0` (the latter with caller-supplied work buffers).

Decided
-------

*   D2.

Questions
---------

*   Add `fwdu` and `invu` for an orthonormal DCT?
