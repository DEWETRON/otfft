Real FFT
========

`RealFFT`: out-of-place transform from `double` samples to `complex_t`
spectrum, and back.

Rules
-----

*   **RFFT-1** `[built]` `fwd*(x, y)` reads real `x` and writes the spectrum
    to `y`. `x` is not changed.
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
