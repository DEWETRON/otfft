Complex FFT
===========

`ComplexFFT`: in-place discrete Fourier transform on `complex_vector`
(`complex_t`: two doubles, `Re` and `Im`).

Rules
-----

*   **CFFT-1** `[done]` Forward transforms replace `x` with its spectrum.
    A delta input gives a flat spectrum, a constant input gives a delta.
    (`otfft_transform_test: TestDeltaSpectrum, TestConstSpectrum`)
*   **CFFT-2** `[done]` Inverse transforms replace a spectrum with the signal.
    (`TestDeltaInverse, TestConstInverse`)
*   **CFFT-3** `[done]` Forward followed by inverse gives back the input, when
    the normalisations of the two match. (`TestIdentity`)
*   **CFFT-4** `[built]` Normalisation per method:

    | Method        | Direction | Scale     |
    | ------------- | --------- | --------- |
    | `fwd`, `fwdn` | forward   | 1/N       |
    | `fwd0`        | forward   | none      |
    | `fwdu`        | forward   | 1/sqrt(N) |
    | `inv`, `inv0` | inverse   | none      |
    | `invu`        | inverse   | 1/sqrt(N) |
    | `invn`        | inverse   | 1/N       |

*   **CFFT-5** `[done]` Any length N works, power of two or not (mixed radix).
    (`otfft_transform_test`, `otfft_optimization_test: TestComplex`)
*   **CFFT-7** `[done]` A length with a prime factor above 100 (for example a
    prime N) is computed with Bluestein's algorithm, also after `setup(n)` on
    an existing object, so it takes O(N log N) time and is correct for any N
    (issues #16 and #17). Results and normalisation are the same as for the
    other lengths. (`otfft_prime_test: TestComplexPrimeSmall,
    TestComplexLargePrimeFactor, TestComplexPrimeAboveIntOverflow,
    TestComplexPrimeIsFast, TestComplexSetupChangesPath`)
*   **CFFT-8** `[done]` For a power-of-two length the algorithm is chosen in
    `setup(n)` from a table per length. There are two tables: one for a
    configured OpenMP thread count of 1 (tuned with `tools/ffttune` on an
    Intel and an AMD CPU) and one for more than 1 thread (the upstream table,
    not re-tuned). The thread count is read when the plan is made, so a plan
    keeps its algorithm if `omp_set_num_threads` is called later. The result
    does not depend on the table. (`otfft_algorithm_table_test:
    TestPowerOfTwoOneThread, TestPowerOfTwoSeveralThreads,
    TestPlanSurvivesThreadCountChange`)
*   **CFFT-9** `[done]` Lengths with a prime factor 7 (together with other small
    factors, or alone) are correct and as fast per point as lengths with 3 and 5.
    (`otfft_radix7_test: TestRadix7Only, TestRadix7WithOtherFactors,
    TestRadix7AboveOpenMpThreshold`)
*   **CFFT-6** `[built]` No leaks or out-of-bounds access for lengths up to 2^24.
    (`TestValgrind` exists, but is compiled only with
    `VALGRIND_MEM_LEAK_DETECTION`, which CI does not set.)

Seen in upstream
----------------

*   `fwd` scales by 1/N and `inv` does not: upstream's convention, kept.
    The doc comment of `fwd` says so; `fwdn` repeats it.

Decided
-------

*   D4: keep upstream's normalisation names.
*   D13: lengths with a prime factor above 100 use Bluestein.
*   D14: two algorithm tables, chosen by the thread count.
*   D15: radix-7 butterfly in the symmetric form (speed only, no change of results).

Questions
---------

*   `fwd` and `fwdn` are the same operation, and so are `inv` and `inv0`.
    Keep both for source compatibility?
