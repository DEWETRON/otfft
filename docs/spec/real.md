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
