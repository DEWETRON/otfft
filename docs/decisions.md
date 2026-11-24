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
