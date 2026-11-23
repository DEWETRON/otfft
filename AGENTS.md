Rules for agents in this repository
===================================

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as in
RFC 2119 and RFC 8174. When you depart from a SHOULD, say so and why in the
pull request.

Changing code
-------------

1.  You MUST NOT change algorithm code in `src/` and `inc/` without a test in
    `unit_tests/` that covers the change.
2.  You SHOULD keep changes small and, for upstream syncs, one upstream
    change per pull request.
3.  You MUST NOT edit `otfft_config.h`; it is generated from
    `src/otfft_config.h.in`.
4.  You MUST keep the public API in `inc/otfft.h` source compatible, or record
    the break in `docs/decisions.md`.

Docs
----

5.  When behaviour changes, you MUST update `docs/spec/` in the same pull
    request, including the status tag.
6.  A new design choice MUST get a new `Dn` entry at the end of
    `docs/decisions.md`. Replaced entries are marked superseded, not deleted.

Tests and pull requests
-----------------------

7.  You MUST run the build and `ctest` (see below) before you say work is done.
8.  You MUST NOT commit or push unless asked.
9.  You MUST NOT add a `Co-Authored-By:` line, or any other agent attribution
    (such as "Generated with Claude Code"), to commit messages or pull
    request descriptions.

This repository
---------------

OTFFT: a C++11 FFT library (Stockham, SSE2/AVX/AVX2, OpenMP), DEWETRON fork of
OK Ojisan's OTFFT.
Original upstream: <http://wwwa.pikara.ne.jp/okojisan/otfft-en/download.html>

### Layout

*   `inc/`: public headers (`otfft.h` is the API).
*   `src/`: algorithms and per-backend headers.
*   `unit_tests/`: Boost.Test suites.
*   `cmake/`: build helpers. `docs/spec/`: behaviour spec. `docs/decisions.md`.

### Checks

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
(cd build && ctest)
```

Needs Boost.Test (`libboost-test-dev`); CI also uses valgrind.
