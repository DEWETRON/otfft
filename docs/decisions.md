Decisions
=========

Why OTFFT (DEWETRON fork) is built the way it is. Each entry gives the
decision, the reason, and what would make us revisit it. Add new entries at
the end; mark replaced ones as superseded rather than deleting them.

Context
-------

**Goal:** a fast FFT library for measurement software, kept close to upstream
OTFFT (Stockham algorithm, AVX, template metaprogramming) so upstream
improvements can be merged.

Entries were written from the repository history and the code; where a date
or owner is unknown it is left out.

D1: Stay in sync with upstream OTFFT
------------------------------------
*   **What:** upstream (<http://wwwa.pikara.ne.jp/okojisan/otfft-en/download.html>) changes (from 11.5 on) are merged in small pull
    requests (#5, #6, #7).
*   **Why:** upstream holds the algorithms; our value is packaging and tests.
*   **Revisit when:** upstream stops, or our changes outgrow merging.

D2: Abstract interfaces behind a factory
----------------------------------------
*   **What:** `ComplexFFT`, `RealFFT`, `RealDCT` are abstract classes created
    by `OTFFT::Factory`. Smart pointers use a custom deleter.
*   **Why:** the caller needs no knowledge of which CPU backend runs.
*   **Rejected:** upstream's concrete classes only; they tie the caller to one
    instruction set at compile time.

D3: One library per CPU backend
-------------------------------
*   **What:** SSE2, AVX and AVX2 are compiled separately and picked at run
    time (`OPTIMIZED_FFT_AUTO`).
*   **Why:** one binary runs well on old and new CPUs.
*   **Risk:** more build time and larger binaries.

D4: Keep upstream normalisation names
-------------------------------------
*   **What:** `fwd` scales by 1/N, `inv` does not; `0`, `u`, `n` variants as
    upstream.
*   **Why:** existing callers and upstream merges.
*   **Revisit when:** a breaking release is acceptable.

D5: Bluestein for arbitrary lengths
-----------------------------------
*   **What:** `createBluesteinFFT` handles lengths the mixed radix does not.
*   **Why:** callers had inputs with prime lengths.

D6: Standard OpenMP
-------------------
*   **What:** use `#pragma omp` from the compiler, not a custom thread layer.
*   **Why:** portable across GCC and Visual Studio.

D7: Specs and decisions live in `docs/`
---------------------------------------
*   **What:** `docs/spec/` holds behaviour rules with ids and status tags;
    this file holds decisions. `AGENTS.md` holds working rules.
*   **Why:** same layout as the other glaure projects.

D8: doctest instead of Boost.Test
---------------------------------
*   **What:** unit tests use doctest 2.4.11, vendored as one header in
    `third_party/doctest`. `BOOST_CHECK_CLOSE` is kept as `CHECK_CLOSE_PCT`
    (same strong percent comparison) in `unit_tests/otfft_test_checks.h`.
*   **Why:** Boost was needed only by the tests. It had to be installed on every
    CI machine, `FindBoost` is gone from CMake 4, and about 80 lines of
    Boost link logic had no caller.
*   **Rejected:** a home-made test macro set (we would have to maintain it).
*   **Done:** CI no longer installs Boost; stale Boost CMake code, `.travis.yml`
    and the duplicate `appyeyor.yml` removed.
*   **Revisit when:** doctest stops being maintained.

D9: GitHub Actions is the only CI
---------------------------------
*   **What:** `.github/workflows/cmake.yml` builds and tests on Ubuntu and
    Windows. AppVeyor (`.appveyor.yml`) and Shippable (`.shippable.yml`) are
    removed, like Travis before.
*   **Why:** the AppVeyor config built only tags on Visual Studio 2013/2017
    images that cannot run our CMake 3.22 minimum; Shippable has shut down.
*   **Revisit when:** we need compilers or platforms GitHub runners lack
    (for example ARM64, roadmap R2.2).

D10: Benchmark against FFTW, PocketFFT and KissFFT
--------------------------------------------------
*   **What:** `benchmarks/` holds a small in-repo harness (no Google Benchmark)
    and adapters for each library. Every result is checked against PocketFFT
    before it is timed. Results go to `docs/benchmarks.md` and
    `docs/benchmark-data/<date>/`.
*   **Why:** roadmap R1.1, and a baseline for the ARM64 and AVX-512 work.
*   **Limit:** FFTW is GPL, so it is found on the system and linked only into
    the benchmark tool; it is never vendored. PocketFFT and KissFFT (BSD) are
    vendored.
*   **Not in CI:** shared runners are too noisy; CI only builds it if asked.
*   **Revisit when:** we want trend tracking, or a quieter benchmark machine.

D11: Benchmark upstream OTFFT through an optional CMake variable
----------------------------------------------------------------
*   **What:** `-DOTFFT_UPSTREAM_DIR=<dir>` builds the sources of an upstream
    OTFFT release as a separate library and a second tool,
    `otfft_bench_upstream`, using the same harness. `benchmarks/compare_upstream.py`
    compares two result files.
*   **Why:** the OTFFT author's published numbers could not be matched, and only
    a same-machine, same-session run of upstream code shows whether this fork
    lost speed. Result: `docs/benchmarks.md`.
*   **Limit:** upstream sources are not vendored; the option is off by default and the
    caller supplies the directory.
*   **Revisit when:** the fork is resynchronised with upstream (R1.8 to R1.11).

D12: ffttune prints the algorithm table, it does not generate library files
---------------------------------------------------------------------------
*   **What:** `tools/ffttune.cpp`, built with `-DOTFFT_BUILD_TOOLS=ON` (off by
    default), is the port of upstream's `ffttune` (roadmap R1.8). It times the
    seven algorithms (DIF4, DIT4, DIF8, DIT8, DIF16, DIT16, SixStep) per
    power-of-two length with the fork's own classes of one backend, checks each
    result against DIF4, and prints a table and the `case` lines for the switch
    in `src/otfft_algo.cpp`. `--out FILE` writes those lines to a file.
*   **Why:** the table decides the speed for large N, and the best choice
    depends on CPU, compiler and thread count (`docs/benchmarks.md`, "Algorithm
    selection baseline"). Upstream's tool writes nine `otfft_gen_*.h` files
    that the library includes; here a person reads the numbers and edits the
    switch, so a table is only changed after it was measured on the machines we
    care about, and in a reviewed pull request with a test.
*   **Rejected:** generated headers included by `src/otfft_algo.cpp` (a tuning
    run on one machine would change the library silently); a run-time tuner
    (planning time and noise, left to R1.8).
*   **Limit:** one backend and one thread count per run; the medians of 9
    batches are not noise-free (the tool marks results within 5% as ties).
    CI builds and runs the tool only if asked.
*   **Revisit when:** the table is re-tuned (R1.8), or run-time tuning is wanted.

D13: createComplexFFT uses Bluestein for lengths with a large prime factor
--------------------------------------------------------------------------
*   **What:** `FFT::setup(n)` (the class behind `createComplexFFT`) builds a
    `Bluestein` object when `n` has a prime factor above 100, and the mixed
    radix code otherwise. The check is in the class, not the factory, so
    `setup(n)` on an existing object also switches. Roadmap R1.4.
*   **Why:** the mixed radix code has butterflies only for 2, 3, 4, 5 and 8;
    any other prime factor k costs O(k) per point, so primes were O(N^2) (48 s at
    N=100,003, issue #17). From k = 46,341 the index `(i*j)%k` overflowed `int`
    and the result was wrong (issue #16). On one Intel notebook the Bluestein
    FFT was as fast or faster from k = 53 to 79 and 2x to 17x faster from
    k = 127; the limit 100 sits between. The index is also computed without
    `%` and without overflow now, in all eight generic loops, so the mixed
    radix code stays correct for any factor.
*   **Rejected:** only fixing the overflow (still O(N^2)); a Bluestein path for
    every length (slower for smooth lengths); a radix-7, 11, 13 butterfly
    (R1.5 is about 7 and smooth lengths, separate work).
*   **Limit:** the real FFT and the DCT of length N call the complex FFT of
    length N/2 directly and are not routed (spec RFFT-6). Factors between 53 and
    79 are close to the crossover; the limit was measured on one CPU and one
    compiler and may move with R1.5.
*   **Revisit when:** R1.5 adds butterflies for more factors, or measurements
    on other CPUs (Ryzen, ARM64) put the crossover elsewhere.

D14: two algorithm tables for the power-of-two FFT, chosen by the thread count
------------------------------------------------------------------------------
*   **What:** `src/otfft_algo.cpp` has two tables of the algorithm per length
    2^1 to 2^24 (DIF4, DIT4, DIF8, DIT8, DIF16, DIT16, SixStep).
    `ALGORITHM_SERIAL` is used when `omp_get_max_threads()` is 1 (or the library
    has no OpenMP) when the plan is made, `ALGORITHM_THREADED` otherwise.
    `ALGORITHM_THREADED` is the table the fork had so far (upstream's);
    `ALGORITHM_SERIAL` differs from it from 2^14. Roadmap R1.8.
*   **Why:** `tools/ffttune` (D12) on an Intel Core Ultra 7 268V (MSVC) and an
    AMD Ryzen 9 9950X (GCC) showed that the old table loses 1.1x to 3.3x with one
    thread from 2^14 (SixStep at 2^16 and above), but that SixStep is the best
    choice with 12 threads (the other algorithms scale poorly: DIF8 at 2^20 is
    2.9x slower). One table cannot serve both, and the two machines also
    disagree on the best one-thread algorithm (Intel DIF8 almost everywhere;
    Ryzen DIF4 at 2^13 to 2^19). `ALGORITHM_SERIAL` takes, for each length from
    2^14, the algorithm with the smallest slowdown on the worse machine (at most
    1.26x of the fastest on either).
*   **Rejected:** one new table for every case (slows threaded runs at large N);
    generated headers (D12); a run-time tuner (R1.8 keeps it as an option).
*   **Limit:** the thread table is not re-tuned (one 12-thread run on one CPU
    only, with odd results at 2^19 and 2^21); the serial table comes from
    two CPUs, one thread, AVX2, and its choices at 2^22 and 2^23 are within
    5% of others. The table is chosen when the plan is made, so changing the
    thread count afterwards does not change the plan. Other backends (SSE2) use
    the same tables.
*   **Revisit when:** a threaded retune exists, or R1.6 (do not thread below the
    break-even size) changes which algorithms scale, or other CPUs (ARM64) are
    measured.

D15: radix-7 butterfly in the symmetric form
--------------------------------------------
*   **What:** `src/otfft_mixedradix.h` has a radix-7 butterfly (`fwdend7`,
    `fwdcore7`, `invend7`, `invcore7` and the OpenMP versions with `p`), called
    when the remaining length is divisible by 7 after 8, 4, 2, 5 and 3. It
    computes the 7-point DFT from the sums and differences of the symmetric
    pairs (x1+x6, x2+x5, x3+x4 and the differences) with real constants (the
    cosines and sines of 2 pi k / 7), 18 multiplications by a real number per
    butterfly instead of 36 complex ones. The twiddle factors of the core
    step are read from the table (`W[k*sp]`, inverse `W[N-k*sp]`), not built
    by repeated multiplication. Roadmap R1.5, issue #18.
*   **Why:** a factor 7 used the generic loop (O(7) complex multiplications per
    output with a table lookup each); N=10,080 was 11x slower than FFTW on the
    Ryzen. Now a factor 7 is 1.9x to 5.5x faster and costs about the same per
    point as 3 and 5.
*   **Rejected:** the same plain form as the radix 3 and 5 butterflies (no
    gain from symmetry); butterflies for 11 and 13 (factors above 7 up to 100
    stay in the generic code, above 100 they use Bluestein, D13).
*   **Limit:** the results differ from the old code in the last bits (other
    order of operations). Only the Intel notebook with MSVC was measured.
*   **Revisit when:** the radix 3 and 5 butterflies are changed to the
    symmetric form (R1.5), or when R2.1 and R2.2 need other kernels.

