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
