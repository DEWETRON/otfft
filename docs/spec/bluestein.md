Bluestein FFT
=============

`createBluesteinFFT` returns a `ComplexFFT` that works for any length,
including primes.

Rules
-----

*   **BLU-1** `[done]` Correct for even lengths. (`otfft_bluestein_test:
    TestBluesteinEven`)
*   **BLU-2** `[done]` Correct for odd lengths. (`TestBluesteinOdd`)
*   **BLU-3** `[done]` Agrees with the mixed-radix FFT where both apply.
    (`otfft_optimization_test: TestBluestein`)
*   **BLU-4** `[built]` Slower than the mixed-radix FFT for lengths that
    factor into small primes; use it only for awkward lengths.
*   **BLU-5** `[built]` Normalisation is that of `complex.md` (CFFT-4).

Seen in upstream
----------------

*   Not in upstream OTFFT; added by this fork.

Decided
-------

*   D5.

Questions
---------

*   Should `createComplexFFT` switch to Bluestein by itself for large prime
    lengths?
