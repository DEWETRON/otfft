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
    in `src/otfft_algo.cpp`. `--out FILE` wrote those lines to a file (replaced by a
    tuning file in D23; the lines are still printed).
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

D16: radix-3 and radix-5 butterflies in the symmetric form
-----------------------------------------------------------
*   **What:** the radix-3 and radix-5 butterflies in `src/otfft_mixedradix.h`
    (`fwdend3`, `fwdcore3`, `fwdend5`, `fwdcore5`, the inverse ones and the
    OpenMP versions with `p`) use the same symmetric form as the radix-7
    butterfly (D15): sums and differences of the symmetric pairs with real
    constants, one `bfly3` and one `bfly5` kernel shared by all of them. The
    twiddle factors of the core step are read from the table (`W[k*sp]`,
    inverse `W[N-k*sp]`). The end functions no longer take the twiddle table.
    Roadmap R1.5, issue #18.
*   **Why:** 2 and 8 real multiplications instead of 4 and 16 complex ones per
    butterfly. Measured on the Intel notebook: lengths with a factor 5 are 1.1x
    to 1.4x faster. Powers of 3 do not change measurably (probably the
    radix-3 pass was already cheap; not investigated), but the code is shared
    with the other radices and not slower beyond noise.
*   **Rejected:** keeping the plain form for radix 3 (no gain, but two forms of
    the same idea in one file); a hand-written radix-9 or radix-25 pass (more
    code for lengths that are rare).
*   **Limit:** the results differ from the old code in the last bits (other
    order of operations). Measured on one CPU, with MSVC and with GCC (WSL2); on
    that CPU the transform is still about 2x slower than FFTW for lengths with
    3, 5 and 7 (docs/benchmarks.md), so the butterflies were not the only cost.
*   **Revisit when:** FFTW comparison shows that the pass structure (number of
    passes, memory traffic) and not the butterflies is the remaining cost, or
    R2.1 and R2.2 need other kernels.

D17: threading thresholds at the measured break-even
----------------------------------------------------
*   **What:** each AVX algorithm class starts to use OpenMP threads from a length
    given by its `OMP_THRESHOLD` (`src/otfft_avx*.h`). New values (log2): DIF4 18,
    DIT4 16, DIF8 16, DIT8 16, DIF16 15, DIT16 14; SixStep stays at 14. With more
    than one thread the threaded table of D14 is used only from 2^15
    (`THREADED_MIN_LOG` in `src/otfft_algo.cpp`), below it the serial table.
    Roadmap R1.6, issue #19.
*   **Why:** measured with `tools/ffttune` (D12) on a Ryzen 9 9950X (12 threads)
    and an Intel Core Ultra 7 268V (2 to 8 threads): the old thresholds (2^11 to
    2^13) made the transform 1.1x to 3.2x slower than with one thread on the Ryzen,
    and the best threaded transform was slower than the best serial one up to 2^14
    (1.5x at 2^13). On the Intel notebook N=16,384 with 4 and 8 threads ran at
    under 1% of the one-thread speed before, and at the one-thread speed after.
*   **Rejected:** one threshold for all classes (on the Ryzen the break-even of the
    classes differs by up to 2^6, from 2^14 to 2^20); thresholds that depend on the thread count or the CPU (the data
    do not show a rule: the break-even of one class changes by 2^4 between two
    CPUs); the Ryzen-only values (2^18 to 2^20 for the radix-4 and radix-8
    classes, which would give up the 2-thread gain on the Intel notebook from
    2^14).
*   **Limit:** a compromise for DIF4, DIT4, DIF8 and DIT8 (2^16 and 2^18,
    from two CPUs); 1.0x to 1.4x losses remain at 2^16 to 2^17 on the Ryzen with
    12 threads. Not run on the Ryzen with the changed library. The thresholds of
    the eight-step, mixed radix, real FFT, DCT and Bluestein code are not
    changed (not measured). The best algorithm with threads still differs
    between the two CPUs (R1.8).
*   **Revisit when:** the threaded table is retuned (R1.8), the Ryzen result is in,
    or a run-time calibration of the break-even is added.

D18: examples check their own result and run under ctest
--------------------------------------------------------
*   **What:** `examples/` has one small program per transform (complex, real,
    DCT, Bluestein). Each one compares its output with a direct formula or a
    round trip, prints the numbers, and returns 0 only if the checks pass.
    `ctest` runs them (`example.<name>`). `OTFFT_BUILD_EXAMPLES` is ON for the
    top-level project and OFF in a subproject. Roadmap R1.3, spec BLD-8.
*   **Why:** an example that nobody runs stops compiling or stops being true
    (for instance the real FFT output length N, not N/2+1, or the scaling of
    `fwd` and `inv`). Running them in the CI that already runs `ctest` costs
    about 10 seconds and needs no new job.
*   **Rejected:** examples that only print (nothing fails when the API
    changes); a separate CI job (no gain); the examples as doctest cases (they
    would not show what a user program looks like).
*   **Limit:** the checks are tolerance checks, not speed checks. The
    examples use the factory API only, with C++11, like the public header.
*   **Revisit when:** a subproject example (a CMake project that uses
    `add_subdirectory(otfft)`) is wanted, or examples for a new backend (R2.1).

D19: exp kernels for the large radix-8 and radix-16 transforms
---------------------------------------------------------------
*   **What:** from N = 2^18 (`EXP_THRESHOLD`, as upstream) the classes DIF8, DIT8, DIF16
    and DIT16 call the exp kernels in `src/otfft_avx{dif,dit}{8,16}exp.h`, ported from
    upstream 11.6e. They compute the twiddle factors in registers (`expj`) and keep no
    weight table (`W = 0` from 2^18). The headers are the upstream ones in the fork's
    namespace, with the entry points `inline`; the body is not re-indented, to keep the
    diff to upstream small. Roadmap R1.9, spec CFFT-12.
*   **Why:** the roadmap asks for them, and the large-N transforms read a table of N/8 to
    N complex numbers from memory in the old path. Measured on the Intel notebook the
    effect is within -7% to +13% for the best class, so the reason to keep them is
    upstream parity and the Ryzen result, not a shown gain (`docs/benchmarks.md`,
    "Exp kernels (R1.9)").
*   **Rejected:** porting `init_Wq` and `init_Wr` (named in the roadmap item; upstream
    11.6e never calls them); changing the algorithm table in the same step (the noise at
    one machine is 5% to 20%, and MSVC and GCC disagree about the best class, so a table
    from this data would be a guess); a different `EXP_THRESHOLD` per class (the same
    data does not say where each class gains: DIF16 gains with GCC and loses with MSVC).
*   **Limit:** no gain shown on the Intel notebook (neutral to +13%); not run on the
    Ryzen. Under GCC the unchanged table now picks DIF8 for 2^18 to 2^21, which is 5% to
    12% slower than before for that class while DIF16 and DIT16 became up to 35% faster.
*   **Revisit when:** the Ryzen result is in; then retune the table (R1.8) and decide a
    threshold per class.

D20: one parallel region for the eight-step code, guided schedule from 2^18
---------------------------------------------------------------------------
*   **What:** in `src/otfft_eightstep.h`, from N = 2^19 (`OMP_THRESHOLD2`) the whole step
    runs in one `omp parallel` region (as upstream 11.6e); below it the eight sub-transforms
    are `omp sections`; the last loop is `nowait`. In `src/otfft_sixstepsq.h` there are two
    thresholds: `OMP_THRESHOLD1 = 2^14` (one thread below), a static schedule from there
    (`part2`) and a guided schedule from `OMP_THRESHOLD2 = 2^17` (`part3`, new), that is
    from 2^18 because the square code exists for even log2 N only. The weight-table set-up in
    `src/otfft_algo_sixstep.cpp` uses `OMP_THRESHOLD1` (same value as before). Roadmap R1.10,
    issue #19, spec CFFT-13.
*   **Why:** the eight-step code called the six-step `part2` outside a parallel region, so its
    `omp for` loops ran on one thread; one region fixes that (SixStep 1.5x to 2.5x faster at
    2^19 to 2^23 with 4 and 8 threads). The guided schedule balances the triangular loops over
    block pairs. See `docs/benchmarks.md`, "OpenMP structure of the eight-step and six-step
    code (R1.10)".
*   **Rejected:** upstream's guided threshold 2^16 (the guided schedule was 0.70x to 0.81x of
    static at N = 2^16 with 8 and 4 threads on the Intel notebook; library speed there did
    not change with the static one); removing the fork's `firstprivate` clauses (not part of
    the problem, kept); changing the algorithm table in the same step (the table is R1.8).
*   **Limit:** one CPU (8 cores, MSVC) with 4 and 8 threads; not run on the Ryzen with 12 or
    16 threads, nor with GCC for speed. With 4 threads the guided schedule is neutral to
    16% slower at 2^18 to 2^20 and with 8 threads up to 1.44x faster. SixStep is still not the
    best class at 2^19 and above on this CPU, so `createComplexFFT` gains only where the table
    selects it.
*   **Revisit when:** the Ryzen result is in (the guided threshold may be lower there), or
    the threaded table is retuned (R1.8).

D21: keep the AVX2 backend, build it with FMA everywhere
--------------------------------------------------------
*   **What:** the AVX2 backend stays. The GCC and Clang flags for it are `-mavx2 -mfma` (CMake);
    the run-time check chooses it only if the CPU has AVX2 and FMA (`CPU_FEATURE_AVX2` in
    `src/otfft.cpp` includes the FMA bit); every backend reports how it was compiled
    (`compiledWithAvx`, `compiledWithAvx2`, `compiledWithFma` in `src/otfft_algo.cpp`) and
    `otfft_backend_test` checks it. Roadmap R1.7, spec FAC-8 and BLD-9.
*   **Why:** the backends share all code and differ in flags. The complex multiplication uses FMA
    where `__FMA__` is defined; MSVC defines it for `/arch:AVX2`, GCC and Clang need `-mfma`.
    Without it the GCC AVX2 backend was up to 3% slower than AVX (the "no gain" of R1.1); with it
    5% to 7% faster over all sizes and 10% to 18% faster at N = 64 to 4,096 (`docs/benchmarks.md`,
    "AVX2 backend: FMA with GCC and Clang (R1.7)").
*   **Rejected:** removing the AVX2 backend (it would give up the MSVC gain of 1.06x to 1.10x and
    this one, to save build time); `-march=haswell` or `-march=native`
    (ties the build to a CPU generation or the build machine); FMA in the AVX or SSE2 backend (an
    AVX-only CPU has no FMA, the code would crash there); a run-time FMA switch inside one
    backend (every call would branch).
*   **Limit:** measured with GCC 15.2, one thread, on one CPU (Core Ultra 7 268V). Not run on the
    Ryzen or with Clang. The Intel compiler flags were not changed. A CPU with AVX2 and no FMA
    (not known to exist in hardware) would use the AVX backend. The dispatch is not unit tested
    (no way to fake CPUID); `otfft_backend_test` checks the build flags only.
*   **Revisit when:** R2.1 adds AVX-512 (the roadmap item asks for a recheck), or the Ryzen shows
    no gain.

D22: run-time tuning file for the algorithm tables
--------------------------------------------------
*   **What:** `OTFFT::loadTuningFile(path, error)` and `OTFFT::resetTuning()` in `inc/otfft.h`.
    A text file (`otfft-tuning 1`, entries `<backend> <serial|threaded> <log2 N> <algorithm>`)
    overrides the built-in tables of `src/otfft_algo.cpp` for the lengths it lists. The store
    is in the common library (`src/otfft_tuning.cpp`), shared by the three backends, and
    guarded by a mutex; `selectAlgorithm` asks it first. Roadmap R1.8, spec FAC-9 to FAC-13.
*   **Why:** the best algorithm differs by machine and compiler (Intel notebook with MSVC, Ryzen
    9 9950X and 3900X with GCC: DIF8 against DIF4 at 2^17 and 2^18, `docs/benchmarks.md`), and
    one built-in table can only be the compromise with the smallest worst slowdown. A file
    lets a machine use its own optimum without a rebuild; the built-in table stays the default.
*   **Rejected:** a tuner inside the library that measures at first use (slow, noisy by a few
    percent, results change between runs); generating source for compilation as upstream's
    `ffttune` does (needs a rebuild per machine); merging a new file into the old one (the
    result would depend on the order of calls); accepting algorithm numbers (names are
    readable and cannot be mistaken for the table's numbering).
*   **Limit:** the file is not checked against the machine (CPU, compiler, thread count), so a
    file from another machine is used as it is. (`ffttune --out` writes this format since D23.)
    No environment variable loads a file. The built-in tables are not changed.
    The tables are per backend and per serial/threaded, not per thread count.
*   **Revisit when:** a wrong file from
    another machine causes trouble (then record the CPU and compiler in the file and check them).

D23: `ffttune --out` writes a tuning file
-----------------------------------------
*   **What:** `--out FILE` writes the format of D22 (spec FAC-10) instead of the `case` lines: the
    `serial` table when `--threads` is 1, the `threaded` table when it is more, one line per
    measured length with the fastest algorithm (ties within 5% are marked in a comment), and
    comments with the CPU model, compiler, backend, thread count and run settings. The
    `case` lines are still printed on standard output. After writing, `ffttune` loads the
    file with `OTFFT::loadTuningFile` and exits with an error if it is rejected;
    `ctest` runs this for both tables (`ffttune.tuning_file`, `ffttune.tuning_file_threaded`).
    Spec BLD-7. Roadmap R1.8.
*   **Why:** one run of the tool gives a file the library can use, with no edit of the source or
    rebuild. The comments record where the file came from, as a person (or later a check) needs
    when a file moves to another machine.
*   **Rejected:** keeping the `case` lines in `--out` and adding a second option for the new
    format (two formats to keep, and the `case` lines are of use only to a person who reads the
    standard output); one run writing both tables (a run measures one thread count).
*   **Limit:** the file is not checked against the machine when loaded (D22). The tool picks the
    fastest algorithm per length; it does not choose a compromise over several machines as the
    built-in tables do (D14). The old `.txt` files in `docs/benchmark-data/` are `case` lines
    and cannot be loaded. The tool is Linux-oriented for the CPU name (`/proc/cpuinfo`; other
    systems write `unknown`).
*   **Revisit when:** a file from the wrong machine is loaded by mistake (check the comment data
    in `loadTuningFile`), or an environment variable loads the file at start.

D24: do not mark the six-step kernels noinline
----------------------------------------------
*   **What:** the six-step kernels (`transpose_kernel` and `mult_twiddle_factor_kernel` in
    `src/otfft_sixstepsq.h`, the eight-step kernels in `src/otfft_eightstep.h`) stay as they are. A patch
    that marks them `noinline` for GCC and Clang is kept in
    `docs/benchmark-data/2026-11-24-ryzen3900x-linux/compile-context/noinline-kernels.patch`.
    Roadmap R1.6, issue #19, `docs/benchmarks.md` ("Why the one-thread speed of the six-step code varies").
*   **Why:** with the kernels inlined into the loops, the six-step transform takes up to 1.8x longer on
    one thread (2^18: 2,500 us against 1,400 us), but only on one thread. With 2 to 12 threads the change
    makes no difference (0.91x to 1.10x), and the default tables use SixStep only with several threads.
    The patch makes 2^15 about 10% slower with 1 thread and 7% with 2.
*   **Rejected:** applying the patch (no gain on any default path, small loss at 2^15); a compiler
    parameter such as more inlining (made it slower, 2,447 us).
*   **Limit:** GCC 15.2.0 on one CPU. MSVC and Clang were not measured.
*   **Revisit when:** a default path uses SixStep with one thread, or the cost of moving data between
    cores falls (for example with the few-exchanges algorithm) so that the kernels matter at several threads.

D25: DIF4 at 2^16 and 2^17 in the threaded table
------------------------------------------------
*   **What:** `ALGORITHM_THREADED` in `src/otfft_algo.cpp` picks DIF4 at 2^16 and 2^17 (was
    SixStep and DIT8, the upstream values). All other entries are unchanged. Roadmap R1.8, spec CFFT-8.
*   **Why:** `ffttune --backend avx2 --threads T`, two runs each for T = 2, 4, 8, 12 and 24 on the
    Ryzen 9 3900X (GCC 15.2.0): DIF4 was the fastest class at 2^16 and 2^17 in every run with 4 or more
    threads. The earlier 12-thread A/B run (`docs/benchmarks.md`) measured the old picks 1.67x (2^16)
    and 2.51x (2^17) slower than DIF4.
*   **Rejected:** also changing 2^15 (the winner changes with the thread count: SixStep, DIF4, DIT4,
    DIF8) and 2^19, 2^20 and 2^24 (the winner changes between runs or the classes are within 5%).
*   **Limit:** one CPU, one compiler, AVX2. Raw runs: `docs/benchmark-data/2026-11-24-ryzen3900x-linux/threaded-table/` (`tT-rN.txt`, T threads, run N). With 2 threads the winner at 2^16 was DIT4 and at 2^17
    DIT16 or DIT4, so DIF4 may lose a little there. The Intel notebook and the 9950X were not run
    with this change. The tuning file (D22) overrides the table on a machine where it is wrong.
*   **Revisit when:** another CPU or compiler picks a different class at these lengths, or a
    threaded retune of the whole table is done.

D26: keep the `11.4xv` version strings, state the sync in the README
--------------------------------------------------------------------
*   **What:** the version line in the file headers (`Version 11.4xv`) is not changed, except in
    the files ported from 11.6e, which already say `11.6e` (`otfft_avx{dif,dit}{8,16}exp.h`).
    The README "Status" says what comes from 11.6e and what does not. Roadmap R1.11.
*   **Why:** compared with upstream 11.6e (whitespace ignored, version lines removed) every file in
    `src/` and `inc/` that has a counterpart differs, from about 100 changed words
    (`otfft_avxdit16omp.h`, `otfft_sixstepsq.h`, `otfft_eightstep.h`) to several thousand
    (`otfft_mixedradix.h`, `otfft_avxdit8.h`). A header that says `11.6e` would claim an identity
    that does not exist. `11.4xv` names the upstream version the file was taken from.
*   **Rejected:** setting all headers to `11.6e` (false); a fork version scheme in the headers
    (nothing reads it, and the library version is in CMake).
*   **Limit:** a file with ported parts from 11.6e (the six-step and eight-step files, R1.10) still
    says `11.4xv`. 11.5e could not be downloaded (404), so which changes came in 11.5e is not known.
*   **Revisit when:** the fork syncs a whole file from a newer upstream release.

D27: do not take over the C99 and Fortran 2003 interfaces and the upstream benchmark programs
---------------------------------------------------------------------------------------------
*   **What:** from upstream 11.6e the fork takes over neither the C99 interface (`otfft_c.h`,
    `otfft_c.cpp`, directory `C99/`), nor the Fortran 2003 interface (`otfft.f03`, directory
    `Fortran2003/`), nor the benchmark and check programs (`fftbench1` to `fftbench5`, `fftcheck`,
    `fftcheck2`, `bstcheck`, `dctcheck`). Roadmap R1.13.
*   **Why:** both interfaces wrap upstream's classes (`FFT0`, `RFFT`, `DCT`, `Bluestein`), which are
    internal in the fork (README); a port is new code on top of `OTFFT::Factory`, with its own
    tests and build, and no user, issue or roadmap item asks for it. The benchmark programs are
    covered: `benchmarks/` compares with FFTW, PocketFFT and KissFFT, and builds upstream OTFFT in
    its own executable (`benchmarks/bench.cpp`); the checks are covered by the doctest suites in
    `unit_tests/`.
*   **Rejected:** copying the C99 and Fortran files as they are (they call upstream's classes, which the
    fork does not export, and would drift from the library); porting `fftbench1` to `fftbench5`
    (`otfft_bench` already covers the comparison with other libraries; upstream's README says
    `fftbench5` needs Ooura's `fftsg.c`, which is not vendored).
*   **Limit:** a C or Fortran caller has no interface. A C wrapper is small if needed (the factory
    functions return abstract classes, so it needs opaque handles and `extern "C"` functions).
*   **Revisit when:** a user needs the library from C or Fortran.

D28: AVX versions of the radix-3, 5 and 7 core passes of the mixed radix code
-----------------------------------------------------------------------------
*   **What:** `fwdcore3`, `fwdcore5`, `fwdcore7`, `invcore3`, `invcore5`, `invcore7` and their OpenMP
    versions (`...p`) in `src/otfft_mixedradix.h` call a 256-bit version (`corePassY`,
    `corePassYp`) when `s` is even and the file is compiled with AVX (`OTFFT_MIXED_YMM`). The
    version handles two values of `q` at once, with the same butterflies operation by operation.
    The SSE2 backend and passes with an odd `s` are unchanged. Roadmap R1.5, spec CFFT-14.
*   **Why:** the passes of radix 3, 5 and 7 were two thirds of the time and worked on one complex
    number per register (`docs/benchmarks.md`, "Where the time goes in the mixed radix code"). Per
    pass the gain is 1.9x to 2.2x; for the whole forward transform, one thread, Ryzen 9 3900X, GCC 15.2.0,
    AVX2: 1.15x (N = 480) to 1.37x (N = 3,000), 1.27x at N = 10,080, 1.34x at N = 100,000, five
    alternating rounds each (`docs/benchmarks.md`, "AVX passes in the library").
*   **Rejected:** a separate set of functions with their own dispatch (the check `s` even is in
    the pass, so `fwdfft` is unchanged); wider registers (AVX-512, R2.1); changing the SSE2 backend
    (it has no 256-bit registers).
*   **Limit:** the end passes (`...end3`, `...end5`, `...end7`, the last pass), the radix-2, 4 and
    8 passes and the first pass (`s` = 1) are still scalar; N = 1,000,000 did not change (0.97x
    to 1.01x, memory bound, one thread). One CPU, one compiler. The AVX passes are not checked
    against the scalar passes of the same build, but against the Bluestein transform (the test
    breaks when either butterfly is changed: a sign flipped in the forward radix 3, 5 or 7
    butterfly fails the test, checked by hand).
*   **Revisit when:** the end passes and the first pass are vectorised, or AVX-512 is added.

D29: AVX version of the first pass (s = 1) of radix 8 and 4 of the mixed radix code
-----------------------------------------------------------------------------------
*   **What:** the `s == 1` branch of `fwdcore8`, `invcore8`, `fwdcore4`, `invcore4` and their OpenMP
    versions in `src/otfft_mixedradix.h` works on two values of `p` at once (`core8s1Y`, `core4s1Y`,
    `OTFFT_MIXED_YMM`). The inputs of `p` and `p+1` are adjacent, the two halves of a result go to
    two places 8 (or 4) points apart. With an odd number of `p` the last one uses the scalar code.
    The twiddle factors w1 to w3 are loaded from the table, w4 to w7 are products (as in the
    `s >= 2` branch). Roadmap R1.5, spec CFFT-14, follows D28.
*   **Why:** the first pass was 28% of the passes at N = 10,080 and cost 1.35 ns per point against
    0.3 to 0.5 for the same radix with `s` = 8 (`docs/benchmarks.md`). One thread, Ryzen 9 3900X,
    GCC 15.2.0, AVX2, one forward transform, 5 alternating rounds against the library of D28:
    1.23x (N = 480), 1.20x (1,000), 1.17x (3,000), 1.16x (10,080), 1.11x (100,000), 0.99x
    (1,000,000).
*   **Rejected:** all seven twiddle factors from the table: faster for small N (1.31x, 1.26x, 1.23x,
    1.19x, 1.10x at the same lengths) but 0.93x at N = 1,000,000, where the pass is memory bound
    and the table is read in seven strided streams; products of w1 only (the same arithmetic as the
    scalar code, bit-identical): 1.40x for the pass alone against 1.7x to 2.0x for the table
    variant (prototype, `docs/benchmark-data/2026-11-25-ryzen3900x-linux/mixed-radix-first-pass/`);
    radix 2 and radix 3, 5, 7 with `s` = 1 (the radix-2 first pass costs 0.39 ns per point, and with
    an odd `s` all passes of the radix 3, 5, 7 stay scalar: a different change, over `p`).
*   **Limit:** results are not bit-identical to the scalar first pass (w1 to w3 from the table, not
    products; in the prototype with all seven twiddle factors from the table the differences to the old result
    were up to 3e-14 absolute, not measured for the kept variant; the table values are the more
    accurate ones). Lengths with an odd part only (no factor 2) and the first pass of radix 2
    are unchanged. One CPU, one compiler, AVX2. The test breaks when the butterfly of any of the
    four functions is changed (a changed add/sub in the first output of each, checked by hand).
*   **Revisit when:** the end passes or the odd-`s` passes are vectorised, or AVX-512 is added.

D30: OpenMP thresholds of the mixed radix code, the real FFT and the DCT
------------------------------------------------------------------------
*   **What:** `OMP_THRESHOLD` of the mixed radix code (`src/otfft_mixedradix.h`) is 2^19 (was 2^15),
    `OMP_THRESHOLD` of `RFFT` and `DCT0` (`src/otfft_base.h`) is 2^20 (was 2^15). The weight tables keep
    their thresholds (`OMP_THRESHOLD_W`), Bluestein is unchanged. Below the threshold the transform runs
    on one thread whatever the thread count. Roadmap R1.6, issue #19, spec CFFT-15, RFFT-7, DCT-6.
*   **Why:** with 12 threads on the Ryzen 9 3900X the mixed radix transform was 0.52x to 0.76x of the
    one-thread speed from 2^15 to about 400,000, and the real FFT and the DCT 0.45x to 0.8x from 2^15
    to 2^18 (`docs/benchmarks.md`, "OpenMP break-even of the mixed radix code, the real FFT and the DCT").
    With the new values the 12-thread time at these lengths is the time of the serial code (0.98x to 1.45x of the
    best one-thread time of the first sweep; the code is serial, so the differences are the placement
    of the buffers), and above them the gains are as before (mixed radix 1.5x to
    1.8x from N = 786,432, real FFT 1.2x to 2.4x, DCT 1.4x to 1.8x from 2^20).
*   **Rejected:** a threshold per thread count or CPU (D17: the break-even of one class changed by
    2^4 between two CPUs, and the plan does not know the CPU); 2^19 for the real FFT and the DCT (the real
    FFT is neutral there; the DCT with the old threshold at 2^19 was 1.13x with 12 threads and 0.86x with 2);
    raising the thresholds of Bluestein and the power-of-two classes (no loss measured).
*   **Limit:** one CPU, one compiler. The Intel notebook and the 9950X were not measured; the values may
    cost a gain there where the break-even is lower. At N = 2^19 the DCT with 12 threads is 7% slower than
    with the old threshold; with 2 threads the real FFT of N = 2^19 is 0.75x of one thread because of
    the SixStep class at 2^18 (not changed here). The tests that exercised the OpenMP versions of the mixed
    radix passes at N >= 2^15 now use lengths from 2^19 (the older lengths stay as serial tests).
*   **Revisit when:** another CPU gives a different break-even, or the threaded table is retuned per thread
    count (R1.8).

D31: place the work buffer of the complex FFT 1728 bytes from the input, modulo 4096
-------------------------------------------------------------------------------------
*   **What:** `FFT::workFor(x)` (`src/otfft_algo.cpp`) returns the work buffer moved, inside its
    allocation, so that `(work - x) mod 4096 == 1728` (a multiple of 32 bytes, so the alignment is
    kept). All eight transform functions of `FFT` use it. `FFT::setup` allocates 4096 bytes more
    for the buffer (at most 4064 are used for the shift). The real FFT, the DCT and Bluestein call
    `FFT` and get it too. Roadmap R1.5, spec CFFT-16.
*   **Why:** the speed of the complex transforms depends on the distance between the input and
    the work buffer modulo 4096 bytes (`docs/benchmarks.md`, "Where the work buffer sits"). Over the 64
    offsets that are multiples of 64 bytes, 104 of 151 lengths (power-of-two and mixed radix, 71 chosen
    by hand and 80 random ones) varied by more than 10% between the best and the worst, up to 3.4x
    (offset 0 is among the worst). 1728 was within 1.06x of the best offset for each of the 71
    lengths used to choose it and within 1.07x for each of the 80 other lengths, on a Ryzen 9 3900X.
    With it the worst of four placements got faster by a median of 1.12x and up to 2.6x, and the
    best by a median of 1.1x (mixed radix up to 1.5x), because the fixed offset also beats the best
    of four random placements.
*   **Rejected:** moving the input (not ours: the caller owns it); one offset per length (the plan
    does not know `x`, and the table would be per CPU); allocating the buffer at a fixed page offset
    (the input moves relative to it); changing the loop order of the passes (a code change for each
    class and not needed).
*   **Limit:** measured on one CPU (Ryzen 9 3900X, GCC 15.2.0, AVX2, one thread). The bad offsets
    are those of 4 KB aliasing and cache sets on this CPU; **1728 is not known to be good on
    another CPU** (the Intel notebook was not measured; `docs/benchmark-data/2026-11-25-ryzen3900x-linux/work-buffer-placement/experiment-env-offset.patch`
    and `scanoff.sh` repeat the scan). The twiddle table and the buffers of the real FFT and the DCT are not
    placed: the power-of-two FFT of N = 16,384 still varies by 1.43x between placements and the real FFT
    of N = 16,384 lost its lucky best case (31.5 us, now 40.6 us; the worst case 43.8 to 42.2 us). The cause of the
    sensitivity (4 KB aliasing, L1 cache sets) was not isolated.
*   **Revisit when:** another CPU gives a different best offset (then it would be chosen at run time or
    per backend), or the passes are changed to use streams that do not alias.

D32: an opt-in AVX-512 backend, built from the AVX2 code, never chosen automatically
-------------------------------------------------------------------------------------
*   **What:** `-DOTFFT_WITH_AVX512=ON` (default OFF) adds a fourth backend library `otfft_avx512`, built
    from the same sources as the others with `-mavx2 -mfma -mavx512f -mavx512dq -mavx512vl`
    (`/arch:AVX512` for MSVC) and `OTFFT_NAMESPACE=OTFFT_AVX512` (`src/otfft_avx512.h`).
    `OptimizationType::OPTIMIZED_FFT_AVX512` exists only then. `sysinfo` detects it from CPUID leaf 7
    (AVX512F bit 16, DQ bit 17, VL bit 31), AVX2 with FMA, and XCR0 bits 1, 2, 5, 6 and 7 (the operating system saves
    the SSE, AVX, opmask and ZMM state). A request that the CPU cannot serve throws `std::domain_error`
    like the other backends; `OPTIMIZED_FFT_AUTO` does not consider it. The tuning file accepts the
    backend name `avx512` (its own tables). Roadmap R2.1, spec FAC-14, FAC-15, BLD-10.
*   **Why:** it is the part of R2.1 that can be built and tested on a machine without AVX-512 (the
    Ryzen 9 3900X, where this was written): the build, the detection, the selection and the failure
    path. The 512-bit kernels need AVX-512 hardware to be written and checked, and upstream 11.6e has
    none to port (only helper functions in `otfft_misc.h`).
*   **Rejected:** letting `AUTO` choose it now (the code is the AVX2 code, so there is nothing to gain and
    on some Intel CPUs heavy 512-bit use lowers the clock; to be decided with measurements); writing the
    512-bit kernels without being able to run them; ON by default (a fourth backend library and
    more build time for no benefit yet); support for the Intel compiler (not available to us).
*   **Limit:** **nothing in this backend has run on AVX-512 hardware.** The compiler is free to emit
    EVEX encodings of the 256-bit code; whether that changes the speed, or the results in the last bit,
    is not known. Only GCC 15.2 was used for building; the Clang and MSVC branches are untested. The
    detection reads CPUID and XGETBV directly; it was written from the Intel documentation and has not
    been checked on a CPU where the bits are set (the test uses `__builtin_cpu_supports` as a second opinion on
    GCC and Clang, and fails if the two disagree on such a CPU).
*   **Update 2026-11-26:** the run on a Ryzen 9 9950X exists (GCC 15.2): the whole suite passes, the CPUID
    detection and `__builtin_cpu_supports` agree, and the speed is that of AVX2 within about 15%
    (`docs/benchmarks.md`). FAC-15 is `[done]`. The pinned `otfft_bench` run of the same day (`OTFFT-avx512`) shows
    it a median 3% faster than AVX2 on one thread (up to 20% from 2^18) and equal with 16 threads; that is
    not yet a reason to let `AUTO` choose it. The limits above that concern Clang, MSVC and Intel CPUs stand.
*   **Revisit when:** a run on AVX-512 hardware exists (then FAC-15 can become `[done]` and `AUTO` can be
    decided), or the 512-bit kernels are written.


D33: switch off the instruction sets a backend must not use
-----------------------------------------------------------
*   **What:** on GCC and Clang the SSE2 backend is built with `-mno-avx -msse2`, the AVX backend with
    `-mno-avx2 -mno-fma -mavx`, and the AVX2 backend with `-mno-avx512f -mavx2 -mfma`. Spec BLD-10 is unchanged.
*   **Why:** `-msse2` and `-mavx` only add features. The GCC 15.2 of the Ubuntu 26.04 GitHub runner defines
    `__AVX2__`, `__AVX__` and `__FMA__` with no `-m` flag at all, so CI built the SSE2 and AVX backends with AVX2
    and FMA (found with a diagnostic run that dumped the predefined macros). The backend test of D21 failed on
    Linux CI since 2026-11-24 and passed on every machine whose compiler targets plain x86-64. A backend built
    that way would also fault with an illegal instruction on a CPU without AVX2, which is what the SSE2 backend
    exists for.
*   **Rejected:** setting `-march=x86-64` for everything (the AVX2 and AVX-512 backends need their own flags
    anyway, and the order of `-march` and `-m` flags is easy to get wrong).
*   **Limit:** the Clang and Intel compiler branches use the same or no flags; only GCC was checked. MSVC selects the
    instruction set with `/arch` and is not affected.

D34: build on Windows with clang-cl as well, because MSVC needs 80 minutes
--------------------------------------------------------------------------
*   **What:** the CMake branch for GCC and Clang excludes clang-cl (`AND NOT MSVC`), which takes the MSVC branch
    and adds `-mfma` (AVX2, AVX-512) and `-msse3` (SSE2). The CI has a job "windows-latest, clang-cl"
    (`-T ClangCL`, Clang 22.1.3, `libomp` given by hand). The MSVC job stays. Spec BLD-11.
*   **Why:** the MSVC job of 2026-11-26 took 79 minutes to build (tests: 1 minute); the heaviest files are
    `otfft_algo_mixed.cpp` and `otfft_algo_sixstep.cpp` at 20 to 25 minutes each, once per backend. The clang-cl job
    builds in about 5 minutes and passes all 5 tests (79 cases, the backend flag test included).
*   **Rejected:** replacing the MSVC job right away (MSVC is what Windows users build with, and nothing here shows
    that the code behaves the same under both); `clang++` with Ninja (not needed, clang-cl runs inside the
    existing generator); dropping OpenMP for clang-cl (the thresholds are part of the tests).
*   **Limit:** one run of one clang-cl version; the speed of the code it builds was not measured (the tests
    prove results, not speed); `/favor:INTEL64` is ignored with a warning; the same SSE3 problem exists for
    Clang on Linux with `-msse2` and was not touched; the path of `libomp` belongs to the Visual Studio 18 image
    and will move when the image does.
*   **Revisit when:** the MSVC job should become optional (run on request or on `master` only; done in D36), or the
    speed of the clang-cl build is compared with GCC and MSVC.

D35: a two-phase threaded FFT (TwoPhase) for 2^16 to 2^22, chosen by default with AVX2 and AVX-512
--------------------------------------------------------------------------------------------------
*   **What:** a new algorithm class `TwoPhase` (number 8; `src/otfft_twophase.h`, `src/otfft_algo_twophase.cpp`)
    for the lengths 2^16 to 2^22: N = R x M, the R child FFTs of size M run as strips in cache-resident tiles on
    their own thread (kernels of the DIF8 class), then one combining pass with the twiddle factors and the R-point FFTs
    of strips of columns; one parallel region, one barrier. It is named `TwoPhase` in tuning files and in `ffttune`
    (which has 8 candidates now). The built-in table gives it to the AVX2 and AVX-512 backends with more than one thread
    from 2^18, and from 2^16 with 8 or more threads (spec CFFT-17 to CFFT-19). Roadmap R1.6.
*   **Why:** the plans FFTW prints for 16 threads have this shape, and OTFFT's six-step code needs about 6.6 sweeps
    over memory at N = 4,194,304 against 2.7 for FFTW (estimated from the bandwidth). With it OTFFT needs 0.60 to 0.83 of
    its time from 2^20 for 2 to 16 threads (Ryzen 9 9950X, AVX2). In the prototype the time at 16 threads is 1.0x to 2.1x
    that of FFTW instead of 2.4x to 3.4x (OTFFT without the input copy); the library was not compared with FFTW again
    (`docs/benchmarks.md`, "Two-phase FFT for several threads").
*   **Rejected:** FFTW's approach in detail (its source and codelets are GPL; only the structure that its plans show was
    used, and the code is written from the existing OTFFT kernels); a new set of kernels (the Stockham kernels with
    `n` and `s` already do the strided work); padding the rows of the work array (3% to 15%, but the work buffer has N
    values and the placement of it is fixed by D31); one default for all backends (only AVX2 was measured); lengths
    outside 2^16 to 2^22 (the kernels and the tile sizes were measured only there; 2^23 and 2^24 stay with SixStep).
*   **Limit:** the default rule rests on **one CPU** (a Ryzen 9 9950X) and the AVX2 backend; the AVX-512 backend gets the
    same rule unmeasured, the SSE2 and AVX backends were not measured with TwoPhase at all (the benchmark run was
    interrupted) and keep the old table. Other CPUs may want other thresholds: `ffttune --threads T --out` finds them.
    Only the forward complex transform was timed in the library. At 2^16 and 2^17 with fewer than 8 threads, and at 2^19
    with 6 threads, it is slower or equal. It is about 1.2x to 2.2x slower than the existing code on one thread (prototype),
    so it is never used with one thread. FFTW is still faster with 16 threads at most lengths.
*   **Revisit when:** a measurement on another CPU (3900X, Intel) or with the SSE2, AVX and AVX-512 backends disagrees
    with the rule; or the work buffer can be padded (D31); or the tile movement (the strided 64-byte pieces cost about twice the
    bandwidth bound) is reworked.

D36: the MSVC job runs after a merge and on request, not on pull requests
-------------------------------------------------------------------------
*   **What:** in `.github/workflows/cmake.yml` the MSVC build is its own job (`msvc`, "windows-latest, MSVC") with
    `if: github.event_name != 'pull_request'`: it runs on every push to `master` and when started by hand
    (`workflow_dispatch`: "Run workflow", or `gh workflow run CMake --ref <branch>`). The Linux jobs and the clang-cl job
    (D34) still run on every pull request.
*   **Why:** the MSVC build took 79 and 42 minutes in two runs of 2026-11-26 (about 5 minutes with clang-cl), so every
    pull request waited up to an hour and a quarter for the slowest job. Nothing requires the check (the repository
    has no branch protection or ruleset).
*   **Rejected:** dropping the MSVC build (MSVC is what Windows users build with and code that only MSVC rejects must
    still be found); running it on pull requests that touch some paths (a path list that misses a case gives false
    confidence; the build breaks on headers that most pull requests touch); a `labeled` trigger (every label change
    would start all jobs again); a matrix entry with skipped steps (it would report success without testing anything).
*   **Limit:** a change that only MSVC rejects is found after the merge, not before. The step is cheap to repeat: start
    the job by hand on the branch before merging when the change touches Windows-specific code, intrinsics or the
    CMake flags of the MSVC branch. The job is not required, so a merge does not wait for it, and a failure on
    `master` has to be fixed by a follow-up pull request.
*   **Revisit when:** MSVC builds get much faster, an MSVC-only break reaches `master` more than once (then run it on pull
    requests again), or a required-checks rule is added to the repository (it would need the job name to be listed).
