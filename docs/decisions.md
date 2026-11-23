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
