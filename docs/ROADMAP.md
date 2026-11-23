Roadmap
=======

What we plan to do next. Items have a stable id (`R<milestone>.<n>`). Mark
finished items with `[x]` and keep them; do not renumber. Behaviour changes
also go in `docs/spec/`, design choices in `docs/decisions.md`.

Milestone 1: Foundations
------------------------

*   [ ] **R1.1** Run benchmarks: measure the transforms (complex, real, DCT,
    Bluestein) per backend and length, and record results in
    `docs/benchmarks.md`.
*   [ ] **R1.2** Remove dependencies: list what the library and the build
    need (for example Boost.Test for the unit tests) and drop or replace
    what is not essential.
*   [ ] **R1.3** Add examples: small buildable programs for the complex,
    real, DCT and Bluestein transforms, built by CMake.

Milestone 2: Platforms
----------------------

*   [ ] **R2.1** Support newer AMD64 instruction set extensions (beyond
    SSE2, AVX and AVX2, for example AVX-512), selected at run time like the
    existing backends (D3).
*   [ ] **R2.2** Support ARM64 (NEON), including a CI job and the same
    run-time backend choice.

Notes
-----

*   R1.1 comes first: it gives the baseline R2.1 and R2.2 are judged against.
