Real FFT
========

`RealFFT`: out-of-place transform from `double` samples to `complex_t`
spectrum, and back.

Rules
-----

*   **RFFT-1** `[built]` `fwd*(x, y)` reads real `x` (N values) and writes the
    spectrum to `y`. `x` is not changed. `y` must hold **N** `complex_t`
    values: the call writes the whole conjugate-symmetric spectrum, the upper
    half mirroring the lower. An N/2+1 buffer overflows (checked by hand with
    AddressSanitizer on 2026-11-23; no unit test yet).
*   **RFFT-2** `[built]` `inv*(x, y)` reads spectrum `x` and writes real `y`.
    **The input `x` is destroyed.**
*   **RFFT-3** `[built]` The length N must be even.
*   **RFFT-4** `[done]` Results match the complex FFT of the same data for
    power-of-two and mixed-radix N.
    (`otfft_optimization_test: TestReal, TestRealMixedRadix`)
*   **RFFT-5** `[built]` Normalisation names and scales are those of
    `complex.md` (CFFT-4).
*   **RFFT-6** `[gap]` The real FFT of length N runs a complex FFT of length
    N/2 that is not routed to Bluestein (CFFT-7), so if N/2 has a prime factor
    above 100 it is slow (O(N^2/4)). It is correct, also above the old int
    overflow limit of issue #16. Right: as fast as the complex FFT.
    (`otfft_prime_test: TestRealPrimeHalfLengthAboveIntOverflow` checks the
    result only.) The DCT does the same.

Seen in upstream
----------------

*   `RFFT` has the same names and the same destroyed-input behaviour.

Decided
-------

*   D2.

Questions
---------

*   What happens for odd N? The code is not documented for it; either reject
    it or fall back to Bluestein.
