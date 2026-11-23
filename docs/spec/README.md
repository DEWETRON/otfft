Behaviour spec
==============

What OTFFT should do, in our own words. It describes what a caller sees and
gets from the public API in `inc/otfft.h`, not how the code is organised.
OTFFT itself is by OK Ojisan (Takuya OKAHISA); the original upstream is at
<http://wwwa.pikara.ne.jp/okojisan/otfft-en/download.html>.
This fork adds multi-CPU
builds, unit tests and OpenMP.

How to read it
--------------

Each rule has an id (`AREA-n`) and a status tag:

*   `[done]`: built, and covered by a test in `unit_tests/`. The test is named.
*   `[built]`: built, but known only from reading the code. No test checks it.
*   `[gap]`: the code does something else or something poorer. The rule says
    what is right.
*   `[planned]`: decided, not built.

Each area file ends with three lists: **Seen in upstream** (behaviour of the
original OTFFT we keep), **Decided** (links to `docs/decisions.md`) and
**Questions**.

How to change it
----------------

*   Describe what the caller sees and gets, not how the code works.
*   A rule about upstream OTFFT goes in only after someone has read or run it.
    Do not guess.
*   When a rule's status changes, change it in the same pull request as the
    code or test.

Files
-----

| File           | Area                                                         |
| -------------- | ------------------------------------------------------------ |
| `factory.md`   | Creating transforms, choosing the CPU backend, lifetime      |
| `complex.md`   | Complex in-place FFT: forward, inverse, normalisation        |
| `real.md`      | Real-to-complex FFT and its inverse                          |
| `dct.md`       | DCT-II and its inverse                                       |
| `bluestein.md` | FFT of any length through Bluestein's algorithm              |
| `build.md`     | Build options, CPU backends, platforms, tests, CI            |

Not written yet: legacy classes `FFT`, `FFT0`, `RFFT`, `DCT`, `DCT0` from
`otfft_fwd.h`, memory helpers (`simd_malloc`), peak finding helpers.
