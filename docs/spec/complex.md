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
    Intel and an AMD CPU) and one for more than 1 thread (the upstream table
    except 2^16 and 2^17, which are DIF4, D25), used from length 2^15 up; below 2^15 the serial table is
    used also with several threads. The thread count is read when the plan is made, so a plan
    keeps its algorithm if `omp_set_num_threads` is called later. The result
    does not depend on the table. (`otfft_algorithm_table_test:
    TestPowerOfTwoOneThread, TestPowerOfTwoSeveralThreads,
    TestPlanSurvivesThreadCountChange, TestThreadedLengthsWithDif4`)
*   **CFFT-9** `[done]` Lengths with a prime factor 7 (together with other small
    factors, or alone) are correct and as fast per point as lengths with 3 and 5.
    (`otfft_radix7_test: TestRadix7Only, TestRadix7WithOtherFactors,
    TestRadix7AboveOpenMpThreshold`)
*   **CFFT-10** `[done]` Lengths with the prime factors 3 and 5, alone or
    together with 2, 4, 8 and 7, are correct, also above the OpenMP threshold.
    (`otfft_radix35_test: TestRadix3Only, TestRadix5Only,
    TestRadix3And5WithOtherFactors, TestRadix3And5AboveOpenMpThreshold`)
*   **CFFT-11** `[built]` A power-of-two transform runs on one thread below a
    length that depends on the algorithm (2^14 to 2^18, D17) and on several above
    it; the lengths are where threads started to pay off in measurements on two
    CPUs. Results do not depend on the thread count. (`otfft_algorithm_table_test`
    checks the results with 1 and 4 threads; the speed is measured in
    `docs/benchmarks.md`, not by a test.)
*   **CFFT-12** `[done]` From length 2^18 the DIF8, DIT8, DIF16 and DIT16 classes of
    every backend (SSE2, AVX, AVX2) compute the twiddle factors in registers instead of
    reading a weight table (the exp kernels of upstream 11.6e, D19). The result is the
    same as for the table path: all six transform functions (`fwd`, `fwd0`, `fwdu`, `inv`,
    `invu`, `invn`) match the direct sum for lengths 2^18 to 2^21 with one and with four
    threads. (`otfft_exp_test: TestExpKernelsOneThread, TestExpKernelsFourThreads`)
*   **CFFT-13** `[done]` The SixStep class (six-step code for even log2 N, eight-step code
    for odd) gives the same result with any thread count, for every length from 2^12 to 2^22,
    on every backend and for all six transform functions, in all of its OpenMP variants: one
    thread below 2^14 (six-step) or 2^13 (eight-step), a static schedule from 2^14, a guided
    schedule from 2^18, `omp sections` for the eight-step code from 2^13 and one parallel
    region around the whole step from 2^19 (D20). (`otfft_sixstep_test:
    TestSixStepOneThread, TestSixStepThreeThreads, TestSixStepEightThreads`; with 3 and 8
    threads the eight sections and the guided chunks split unevenly.)
*   **CFFT-14** `[done]` In the AVX and AVX2 backends the radix-3, 5 and 7 core passes of the
    mixed radix code (a factor 3, 5 or 7 that is not the last one) work on two values of q at
    once when s is even, forward and inverse, serial and OpenMP (N >= 2^19, D30), and give the same
    result as the Bluestein transform of the same length, at every point. With an odd s (no
    factor 2 before) and in the SSE2 backend the scalar passes are used. (D28.) The same holds for the
    first pass (s = 1) of radix 8 and radix 4, which works on two values of p at once; with an odd
    number of p the last one uses the scalar code. (D29.)
    (`otfft_mixed_avx_test: TestAvxPassesSerial, TestAvxPassesOddS,
    TestAvxPassesAboveOpenMpThreshold, TestAvxFirstPassSerial, TestAvxFirstPassAboveOpenMpThreshold`)
*   **CFFT-15** `[done]` The mixed radix code runs on one thread below N = 2^19 whatever the
    thread count, and on several threads from there on; the result does not depend on the thread count.
    (D30.) (`otfft_omp_threshold_test: TestMixedRadixAroundThreshold`; the speed is measured in
    `docs/benchmarks.md`, not by a test.)
*   **CFFT-16** `[done]` The work buffer of the complex FFT (every algorithm, and the real FFT, the DCT
    and Bluestein through it) is placed inside its allocation so that its distance to the input is
    1728 bytes modulo 4096, at each call; the plan allocates 4 KB more for it. The result does not depend
    on where the input is. (D31.) (`otfft_placement_test: TestPowerOfTwoAtEveryPageOffset,
    TestMixedRadixAtEveryPageOffset, TestBluesteinPathAtEveryPageOffset,
    TestRealFftAndDctAtEveryPageOffset`: the input at every multiple of 32 bytes inside a page; the
    speed is measured in `docs/benchmarks.md`, not by a test.)
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
*   D16: radix-3 and radix-5 butterflies in the symmetric form (speed only).
*   D17: threading thresholds at the measured break-even.

Questions
---------

*   `fwd` and `fwdn` are the same operation, and so are `inv` and `inv0`.
    Keep both for source compatibility?
