Benchmarks
==========

How fast OTFFT is next to FFTW, PocketFFT and KissFFT, measured on one machine
on 2026-11-23. Roadmap item R1.1. The numbers are indicative, not a
certificate: read "Caveats" before quoting them.

Summary
-------

*   **Complex FFT, powers of two, one thread.** OTFFT is as fast as FFTW up to
    N=64 and within 1.3x up to N=16,384. From N=65,536 FFTW is 1.6x
    to 3.1x faster, except at N=4,194,304 where OTFFT is ahead (and the
    numbers are noisy). OTFFT is 1.3x to 26x faster than PocketFFT and KissFFT
    on these sizes.
*   **DCT-II.** OTFFT is 1.1x to 1.9x faster than FFTW for powers of two up to
    N=262,144.
*   **Real FFT.** FFTW is 1.0x to 1.5x faster up to N=262,144.
*   **Other lengths are weak.** For lengths with factors 3, 5 and 7, OTFFT is
    1.9x to 11x slower than FFTW (10,080 = 2^5 * 3^2 * 5 * 7 is the worst).
    For prime lengths `createComplexFFT` is unusable: 12 MFLOPS at N=1,009 and
    a wrong result at N=100,003 (see "Problems found"). `createBluesteinFFT`
    is the way to do primes: it beats PocketFFT at N=1,009 and 10,007 (2.5x and
    1.2x) but not at N=100,003 (0.7x), and reaches 0.2x to 1.0x of FFTW's
    speed.
*   **Backends.** AVX is 1.6x to 2.2x faster than SSE2 up to N=16,384. AVX2 is
    no faster than AVX. From N=65,536 up all backends are within about 25%.
*   **Threads.** OTFFT with 12 threads is 1.8x to 3.9x faster for N >= 65,536, but
    slower for N=16,384 (0.4x). FFTW scales better and is 1.2x to 4.4x ahead of
    OTFFT with 12 threads for N >= 16,384.
*   **Upstream claim.** The OTFFT author's page says OTFFT beats FFTW at every length from 2^1
    to 2^24. That is not reproduced here, with this fork or with upstream 11.6e
    itself, built and run in the same session (see "Same-session comparison
    with upstream 11.6e"). On average the fork is within 5% of upstream below
    N=65,536 and differs by a factor of 0.5 to 1.6 at some larger sizes.
*   **Compilers.** GCC 15.2 and Clang 21.1.8 give the same picture; the real FFT and DCT
    are 4 to 12 percent slower with Clang and Bluestein 15 to 43 percent. See "Compiler comparison".
*   **Other CPU.** On an Intel Core Ultra 7 268V with MSVC (no FFTW) OTFFT is
    the fastest of OTFFT, PocketFFT and KissFFT at every power of two and
    shows the same weak spots (N=10,080, prime lengths). See "Intel Core Ultra
    7 268V, Windows, MSVC".
*   **Planning.** OTFFT plans in 8 microseconds to 100 ms. FFTW with
    `FFTW_MEASURE` needs 0.3 ms to 10 s (the time limit we set).

Setup
-----

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 9 3900X, 12 cores / 24 threads |
| OS | Linux 7.0, CPU governor `schedutil` (not `performance`) |
| Compiler | GCC 15.2.0, CMake build type Release |
| OTFFT | commit `070e1bd` (master), backends SSE2, AVX and AVX2, `OPTIMIZED_FFT_AUTO` unless named |
| FFTW | 3.3.10, built from the official tarball with SSE2, AVX, AVX2 and OpenMP; `FFTW_MEASURE` and `FFTW_ESTIMATE`, planning time limit 10 s |
| PocketFFT | C++ header-only, commit `4d865cb` (branch `cpp`), vendored in `third_party/pocketfft` |
| KissFFT | commit `de93ec3`, double precision, vendored in `third_party/kissfft` |
| Machine state | A shared development machine, not isolated or frequency-pinned |

Method
------

*   **What is timed:** one forward, unnormalised transform of a given input
    with the plan already built. OTFFT is called through `fwd0`.
*   **Copying:** OTFFT complex and DCT transforms work in place, so each
    timed run includes copying the input into the work buffer. The other
    libraries (and OTFFT real) work out of place and do not copy.
*   **Flops:** MFLOPS are "equivalent": 5 N log2(N) for complex and 2.5 N log2(N)
    for real and DCT, divided by the measured time (the benchFFT convention).
    Higher is better. They are for comparing libraries, not real operation counts.
*   **Statistics:** each case runs batches of about 2 ms, 11 batches (3 if one run takes
    over 200 ms), and reports the median time per run. The minimum is in the CSV.
*   **Correctness first:** every result is compared with PocketFFT before it is
    timed. A library with a relative error above 1e-9 is reported as "wrong"
    and not timed. A case where one run takes over 2 s is "slow".
*   **Data:** random numbers in [-1, 1], aligned to 64 bytes.
*   **Sizes:** powers of two from 16 to 4,194,304; smooth lengths 480, 1,000,
    3,000, 10,080, 100,000 and 1,000,000; primes 1,009, 10,007 and 100,003.
*   **Threads:** OTFFT and FFTW use the given thread count (OpenMP). PocketFFT and
    KissFFT are single-threaded in 1D, so their 12-thread columns repeat the
    one-thread numbers.
*   **Not supported:** OTFFT's real and DCT transforms need an even length, so odd (prime)
    lengths are blank for them. KissFFT has no DCT.

Results: one thread
-------------------

**Complex FFT, powers of two, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 20,453 | 20,841 | **20,846** | 860 | 796 |
| 64 | 28,405 | 30,530 | **30,741** | 2,626 | 2,963 |
| 256 | 28,760 | 37,627 | **37,775** | 5,135 | 5,180 |
| 1,024 | 27,393 | 35,744 | **35,771** | 6,672 | 5,832 |
| 4,096 | 23,500 | **27,446** | 27,356 | 7,409 | 5,914 |
| 16,384 | 17,982 | 19,532 | **19,744** | 7,391 | 5,379 |
| 65,536 | 6,907 | **21,543** | 21,515 | 4,726 | 5,110 |
| 262,144 | 8,185 | **18,010** | 16,624 | 4,196 | 3,901 |
| 1,048,576 | 5,346 | **8,395** | 8,176 | 3,095 | 2,256 |
| 4,194,304 | **5,800** | 3,083 | 4,093 | 2,546 | 1,603 |

**Complex FFT, smooth lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT | OTFFT-bluestein |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 480 | 10,676 | **30,314** | 30,021 | 6,224 | 5,226 | 2,389 |
| 1,000 | 8,036 | **25,948** | 25,893 | 6,583 | 5,789 | 2,630 |
| 3,000 | 7,816 | **27,562** | 27,390 | 7,016 | 5,844 | 1,471 |
| 10,080 | 2,271 | **25,754** | 25,618 | 7,649 | 3,113 | 1,461 |
| 100,000 | 6,452 | 21,674 | **22,274** | 6,943 | 5,398 | 815 |
| 1,000,000 | 4,959 | 9,625 | **9,757** | 4,357 | 1,942 | 739 |

**Complex FFT, prime lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT | OTFFT-bluestein |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,009 | 12 | **2,892** | 2,890 | 1,095 | 42 | 2,775 |
| 10,007 | 2 | 3,855 | **3,949** | 1,189 | 5 | 1,462 |
| 100,003 | wrong | 3,182 | **3,184** | 1,091 | slow | 778 |

**Real FFT (real to complex), powers of two, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7,849 | **8,480** | 8,426 | 496 | 421 |
| 64 | 14,253 | 14,279 | **14,538** | 1,645 | 1,718 |
| 256 | 18,483 | **23,526** | 23,478 | 3,516 | 3,716 |
| 1,024 | 20,497 | 26,753 | **26,860** | 5,853 | 4,786 |
| 4,096 | 20,849 | 25,713 | **25,908** | 7,122 | 5,068 |
| 16,384 | 16,008 | 23,401 | **23,617** | 7,204 | 5,079 |
| 65,536 | 15,406 | 20,478 | **20,751** | 8,125 | 4,693 |
| 262,144 | 15,356 | **19,707** | 19,429 | 7,884 | 4,582 |
| 1,048,576 | 5,580 | 5,898 | **6,220** | 4,390 | 2,388 |
| 4,194,304 | **5,151** | 4,584 | 4,484 | 2,036 | 1,537 |

**Real FFT (real to complex), smooth lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 480 | 8,412 | **24,679** | 24,600 | 4,400 | 4,519 |
| 1,000 | 7,343 | **20,127** | 19,838 | 4,847 | 5,259 |
| 3,000 | 7,144 | 22,365 | **22,454** | 5,646 | 5,464 |
| 10,080 | 2,199 | 22,326 | **22,355** | 5,797 | 3,102 |
| 100,000 | 5,670 | **19,582** | 19,314 | 6,087 | 5,734 |
| 1,000,000 | 4,529 | **7,230** | 7,183 | 4,244 | 2,568 |

**Real FFT (real to complex), prime lengths, MFLOPS**

| N | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: |
| 1,009 | 1,650 | **1,658** | 535 |
| 10,007 | 1,208 | **1,210** | 605 |
| 100,003 | **865** | 862 | 556 |

**DCT-II, powers of two, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: | ---: |
| 16 | **4,543** | 3,705 | 3,697 | 321 |
| 64 | **7,752** | 6,778 | 6,626 | 1,055 |
| 256 | **11,303** | 5,957 | 5,922 | 2,229 |
| 1,024 | **13,369** | 7,293 | 7,252 | 3,623 |
| 4,096 | **13,528** | 7,526 | 7,529 | 4,628 |
| 16,384 | **11,656** | 6,732 | 6,715 | 5,048 |
| 65,536 | **12,193** | 6,321 | 6,316 | 5,424 |
| 262,144 | **10,328** | 5,682 | 5,629 | 5,550 |
| 1,048,576 | 3,784 | 4,748 | **4,759** | 3,402 |
| 4,194,304 | **3,851** | 884 | 886 | 1,656 |

**DCT-II, smooth lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: | ---: |
| 480 | **6,882** | 6,176 | 6,229 | 2,816 |
| 1,000 | 6,238 | **6,408** | 6,408 | 3,419 |
| 3,000 | 6,262 | 6,715 | **6,731** | 4,170 |
| 10,080 | 2,145 | 6,784 | **6,786** | 4,417 |
| 100,000 | 5,362 | 6,100 | **6,124** | 5,254 |
| 1,000,000 | 3,575 | 3,207 | 3,263 | **3,908** |

**DCT-II, prime lengths, MFLOPS**

| N | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: |
| 1,009 | 1,598 | **1,632** | 516 |
| 10,007 | 1,168 | **1,184** | 586 |
| 100,003 | **846** | 845 | 536 |

**OTFFT backends, complex FFT, powers of two, MFLOPS**

| N | OTFFT-sse2 | OTFFT-avx | OTFFT-avx2 | OTFFT |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 12,005 | 20,243 | 20,330 | **20,453** |
| 64 | 13,591 | 27,669 | 28,141 | **28,405** |
| 256 | 13,145 | 28,731 | 28,638 | **28,760** |
| 1,024 | 13,288 | **27,981** | 27,861 | 27,393 |
| 4,096 | 13,410 | 25,288 | **25,324** | 23,500 |
| 16,384 | 11,146 | **18,003** | 17,954 | 17,982 |
| 65,536 | 6,758 | 6,864 | **6,922** | 6,907 |
| 262,144 | 7,508 | 8,188 | **8,217** | 8,185 |
| 1,048,576 | 4,690 | **5,360** | 5,352 | 5,346 |
| 4,194,304 | 4,732 | 5,832 | **5,900** | 5,800 |

**Plan creation, complex FFT, powers of two, microseconds**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 21 | 298 | 4 | 3 | 6 |
| 64 | 8 | 14,539 | 10 | 3 | 2 |
| 256 | 8 | 41,496 | 23 | 4 | 5 |
| 1,024 | 10 | 76,745 | 71 | 11 | 16 |
| 4,096 | 125 | 161,874 | 291 | 38 | 62 |
| 16,384 | 125 | 504,749 | 249 | 200 | 253 |
| 65,536 | 1,164 | 1,705,447 | 1,567 | 1,505 | 1,340 |
| 262,144 | 4,839 | 6,952,789 | 3,109 | 8,876 | 6,108 |
| 1,048,576 | 23,430 | 4,097,820 | 9,506 | 52,989 | 30,644 |
| 4,194,304 | 100,463 | 10,344,833 | 67,576 | 252,540 | 132,953 |

**Wrong results (not timed)**

| Library | Kind | N | Relative error |
| --- | --- | ---: | ---: |
| OTFFT | complex | 100,003 | 167 |
| OTFFT-sse2 | complex | 100,003 | 175 |
| OTFFT-avx | complex | 100,003 | 167 |
| OTFFT-avx2 | complex | 100,003 | 167 |

Note: OTFFT dct output is 0.5 times PocketFFT's (convention only).

Note: OTFFT-avx dct output is 0.5 times PocketFFT's (convention only).

Note: OTFFT-avx2 dct output is 0.5 times PocketFFT's (convention only).

Note: OTFFT-sse2 dct output is 0.5 times PocketFFT's (convention only).

Results: 12 threads
-------------------

FFTW is given 12 threads for every size, so it is slow for small N where thread
start-up costs more than the work. A real application would use one thread
there. Compare the threaded columns for N >= 65,536.

**Complex FFT, powers of two, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 20,428 | 21,311 | **21,314** | 860 | 796 |
| 64 | **28,412** | 24,738 | 24,437 | 2,626 | 2,963 |
| 256 | **28,946** | 1,476 | 1,456 | 5,135 | 5,180 |
| 1,024 | **27,648** | 5,826 | 5,832 | 6,672 | 5,832 |
| 4,096 | **22,663** | 13,500 | 13,697 | 7,409 | 5,914 |
| 16,384 | 7,256 | **32,866** | 31,849 | 7,391 | 5,379 |
| 65,536 | 14,427 | 40,290 | **41,944** | 4,726 | 5,110 |
| 262,144 | 16,901 | 47,809 | **47,953** | 4,196 | 3,901 |
| 1,048,576 | 20,588 | **24,586** | 24,250 | 3,095 | 2,256 |
| 4,194,304 | 10,244 | **17,285** | 17,154 | 2,546 | 1,603 |

**Complex FFT, smooth lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT | OTFFT-bluestein |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 480 | **10,353** | 2,356 | 2,379 | 6,224 | 5,226 | 2,366 |
| 1,000 | **7,896** | 5,191 | 5,219 | 6,583 | 5,789 | 2,610 |
| 3,000 | 7,707 | 11,552 | **12,235** | 7,016 | 5,844 | 660 |
| 10,080 | 2,170 | 23,962 | **24,294** | 7,649 | 3,113 | 805 |
| 100,000 | 9,505 | 54,477 | **54,840** | 6,943 | 5,398 | 2,206 |
| 1,000,000 | 10,042 | **34,998** | 32,798 | 4,357 | 1,942 | 906 |

**Complex FFT, prime lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT | OTFFT-bluestein |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,009 | 12 | 985 | 980 | 1,095 | 42 | **2,633** |
| 10,007 | 2 | 3,168 | **3,191** | 1,189 | 5 | 800 |
| 100,003 | wrong | 5,001 | **5,020** | 1,091 | slow | 2,198 |

**Real FFT (real to complex), powers of two, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 7,522 | **8,247** | 8,144 | 496 | 421 |
| 64 | 13,353 | **13,703** | 13,673 | 1,645 | 1,718 |
| 256 | **17,527** | 15,123 | 15,008 | 3,516 | 3,716 |
| 1,024 | **20,135** | 4,708 | 4,501 | 5,853 | 4,786 |
| 4,096 | **19,565** | 13,728 | 13,861 | 7,122 | 5,068 |
| 16,384 | 6,461 | 23,922 | **23,939** | 7,204 | 5,079 |
| 65,536 | 8,770 | 33,056 | **33,092** | 8,125 | 4,693 |
| 262,144 | 9,140 | 33,946 | **34,034** | 7,884 | 4,582 |
| 1,048,576 | 4,959 | 18,584 | **18,593** | 4,390 | 2,388 |
| 4,194,304 | 5,702 | **12,646** | 12,376 | 2,036 | 1,537 |

**Real FFT (real to complex), smooth lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 480 | **8,364** | 2,764 | 2,818 | 4,400 | 4,519 |
| 1,000 | **7,244** | 5,325 | 5,609 | 4,847 | 5,259 |
| 3,000 | 7,215 | **9,657** | 9,631 | 5,646 | 5,464 |
| 10,080 | 2,154 | 19,358 | **19,982** | 5,797 | 3,102 |
| 100,000 | 6,832 | **37,299** | 35,907 | 6,087 | 5,734 |
| 1,000,000 | 7,558 | **24,300** | 23,375 | 4,244 | 2,568 |

**Real FFT (real to complex), prime lengths, MFLOPS**

| N | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: |
| 1,009 | **754** | 751 | 535 |
| 10,007 | **1,313** | 1,294 | 605 |
| 100,003 | 1,189 | **1,190** | 556 |

**DCT-II, powers of two, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: | ---: |
| 16 | **4,308** | 3,466 | 3,467 | 321 |
| 64 | **7,970** | 6,393 | 6,437 | 1,055 |
| 256 | **11,082** | 906 | 944 | 2,229 |
| 1,024 | **12,856** | 2,729 | 2,729 | 3,623 |
| 4,096 | **13,193** | 4,405 | 4,420 | 4,628 |
| 16,384 | 4,928 | 6,198 | **6,347** | 5,048 |
| 65,536 | 5,200 | **8,154** | 7,920 | 5,424 |
| 262,144 | 5,797 | 11,473 | **11,553** | 5,550 |
| 1,048,576 | 3,972 | **8,622** | 8,479 | 3,402 |
| 4,194,304 | **4,080** | 2,619 | 2,574 | 1,656 |

**DCT-II, smooth lengths, MFLOPS**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: | ---: |
| 480 | **6,896** | 844 | 1,212 | 2,816 |
| 1,000 | **6,240** | 1,521 | 1,527 | 3,419 |
| 3,000 | **6,172** | 3,144 | 3,216 | 4,170 |
| 10,080 | 2,146 | 5,506 | **5,541** | 4,417 |
| 100,000 | 4,683 | 7,082 | **7,440** | 5,254 |
| 1,000,000 | **5,182** | 3,419 | 3,449 | 3,908 |

**DCT-II, prime lengths, MFLOPS**

| N | FFTW-measure | FFTW-estimate | PocketFFT |
| ---: | ---: | ---: | ---: |
| 1,009 | **668** | 661 | 516 |
| 10,007 | 1,152 | **1,202** | 586 |
| 100,003 | **1,173** | 1,132 | 536 |

**OTFFT backends, complex FFT, powers of two, MFLOPS**

| N | OTFFT-sse2 | OTFFT-avx | OTFFT-avx2 | OTFFT |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 12,204 | **20,492** | 14,574 | 20,428 |
| 64 | 13,703 | 28,387 | **28,631** | 28,412 |
| 256 | 13,264 | **29,162** | 29,124 | 28,946 |
| 1,024 | 13,402 | 28,225 | **28,255** | 27,648 |
| 4,096 | 13,360 | **25,116** | 22,195 | 22,663 |
| 16,384 | 6,990 | 7,220 | 7,202 | **7,256** |
| 65,536 | 13,726 | 14,565 | **14,621** | 14,427 |
| 262,144 | 16,584 | **17,663** | 17,315 | 16,901 |
| 1,048,576 | 18,498 | 19,744 | 18,545 | **20,588** |
| 4,194,304 | 9,492 | 10,078 | **10,677** | 10,244 |

**Plan creation, complex FFT, powers of two, microseconds**

| N | OTFFT | FFTW-measure | FFTW-estimate | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 27 | 350 | 5 | 3 | 6 |
| 64 | 10 | 18,641 | 6 | 3 | 2 |
| 256 | 8 | 63,464 | 41 | 4 | 5 |
| 1,024 | 12 | 135,219 | 126 | 11 | 16 |
| 4,096 | 144 | 271,788 | 530 | 38 | 62 |
| 16,384 | 140 | 571,648 | 1,665 | 200 | 253 |
| 65,536 | 914 | 1,264,033 | 1,974 | 1,505 | 1,340 |
| 262,144 | 4,221 | 3,936,919 | 3,264 | 8,876 | 6,108 |
| 1,048,576 | 19,195 | 10,016,911 | 6,483 | 52,989 | 30,644 |
| 4,194,304 | 73,807 | 10,066,233 | 45,577 | 252,540 | 132,953 |

**Wrong results (not timed)**

| Library | Kind | N | Relative error |
| --- | --- | ---: | ---: |
| OTFFT | complex | 100,003 | 167 |
| OTFFT-sse2 | complex | 100,003 | 175 |
| OTFFT-avx | complex | 100,003 | 167 |
| OTFFT-avx2 | complex | 100,003 | 167 |

Note: OTFFT dct output is 0.5 times PocketFFT's (convention only).

Note: OTFFT-avx dct output is 0.5 times PocketFFT's (convention only).

Note: OTFFT-avx2 dct output is 0.5 times PocketFFT's (convention only).

Note: OTFFT-sse2 dct output is 0.5 times PocketFFT's (convention only).

Comparison with the upstream benchmark
---------------------------------------

The OTFFT author publishes benchmarks at
<http://wwwa.pikara.ne.jp/okojisan/otfft-en/index.html> (read on 2026-11-23;
the page says it was last changed 2026-06-03) and states that OTFFT is faster
than FFTW for sequence lengths 2^1 to 2^24. This section sets those numbers
next to a run of this repository.

**What upstream measures:** the summed time of a complex DFT and IDFT (a round
trip) on power-of-two lengths 2^1 to 2^24, double precision, in place for
OTFFT, with the percentage of FFTW's time in brackets. The tables below use
the same idea: `otfft_bench --kinds roundtrip --sizes pow2all` times OTFFT's
`fwd` (which scales by 1/N) followed by `inv` on the same buffer, with no
copying, against FFTW's forward and backward plans out of place. Under 100%
means OTFFT is faster than FFTW. The upstream percentages below are copied
from that page.

**One thread**

| N | Upstream: OTFFT / FFTW | This run: OTFFT / FFTW-measure |
| ---: | ---: | ---: |
| 2^1 | 40% | 105% |
| 2^2 | 44% | 107% |
| 2^3 | 59% | 108% |
| 2^4 | 58% | 95% |
| 2^5 | 62% | 85% |
| 2^6 | 67% | 96% |
| 2^7 | 84% | 105% |
| 2^8 | 64% | 124% |
| 2^9 | 77% | 111% |
| 2^10 | 76% | 129% |
| 2^11 | 87% | 109% |
| 2^12 | 90% | 104% |
| 2^13 | 84% | 120% |
| 2^14 | 84% | 116% |
| 2^15 | 75% | 102% |
| 2^16 | 100% | 303% |
| 2^17 | 107% | 117% |
| 2^18 | 94% | 174% |
| 2^19 | 80% | 123% |
| 2^20 | 85% | 148% |
| 2^21 | 94% | 171% |
| 2^22 | 88% | 30% |
| 2^23 | 85% | 37% |
| 2^24 | 82% | – |
| cost (sum, 2^1 to 2^21) | 84% (2^1 to 2^24) | 158% |

*Upstream: MacBook Pro 2016 (Core i7 2.6 GHz), macOS, Clang 8.0, OTFFT 11.5,
FFTW 3.3.8 with `FFTW_MEASURE` (AVX2). This run: Ryzen 9 3900X, Linux, GCC
15.2, OTFFT at this repository's `master`, FFTW 3.3.10 with `FFTW_MEASURE`.
Sizes 2^22 and 2^23 are not reliable here: FFTW's round trip took 2 to 3 times
longer than its forward-only time suggests (the 10 s planning limit and memory
bandwidth), which is why the cost row stops at 2^21. At 2^24 a single FFTW run
took over 2 s and was skipped.*

**12 threads** (compare from 2^13 up only)

| N | Upstream: OTFFT / FFTW | This run: OTFFT / FFTW-estimate |
| ---: | ---: | ---: |
| 2^1 | 39% | 113% |
| 2^2 | 49% | 102% |
| 2^3 | 56% | 110% |
| 2^4 | 66% | 97% |
| 2^5 | 76% | 90% |
| 2^6 | 86% | 73% |
| 2^7 | 91% | 59% |
| 2^8 | 99% | 5% |
| 2^9 | 91% | 9% |
| 2^10 | 81% | 18% |
| 2^11 | 89% | 27% |
| 2^12 | 92% | 49% |
| 2^13 | 85% | 224% |
| 2^14 | 73% | 293% |
| 2^15 | 49% | 197% |
| 2^16 | 23% | 135% |
| 2^17 | 38% | 302% |
| 2^18 | 43% | 132% |
| 2^19 | 46% | 462% |
| 2^20 | 80% | 69% |
| 2^21 | 87% | 280% |
| 2^22 | 92% | 141% |
| 2^23 | 86% | 288% |
| 2^24 | 83% | 91% |

*Upstream: Surface Laptop 5 (10 cores / 12 threads), Ubuntu on WSL2, GCC 11.4,
OTFFT 11.6, FFTW 3.3.10 with `FFTW_ESTIMATE`. This run: 12 threads for both
libraries. Up to 2^12 the figures here are meaningless: FFTW was forced to use
12 threads for transforms where thread start-up costs far more than the work,
whereas upstream's FFTW times for those sizes (for example 0.64 microseconds at
2^8) show it ran on one thread there. Above 2^12 the run was also noisy
(OTFFT's time at 2^19 is higher than at 2^20), so no cost row is given.*

**What this shows**

*   The claim "OTFFT is faster than FFTW" is **not reproduced** on this
    machine. With one thread OTFFT and FFTW are within about 10% of each other
    up to 2^7 (within 15%) and FFTW is 1.0 to 1.3 times faster from 2^8 to 2^15
    (about 1.2 to 3 times from 2^16 to 2^21). Upstream reports OTFFT 10% to 60%
    faster on most of these sizes.
*   With 12 threads FFTW is ahead from 2^13 on in all but two of the sizes, in
    contrast to upstream's table where OTFFT is 8% to 77% faster.
*   The forward-only tables earlier in this document agree with the round trip
    results. Their OTFFT numbers include a copy of the input, which the round
    trip avoids, and the picture is about the same.
*   Possible reasons: a different CPU (Zen 2 against a 2016 Intel core), compiler
    (GCC 15.2 against Clang 8) and FFTW version (3.3.10 against 3.3.8). Upstream's
    own code was run on this machine too, see the next section: it does not beat
    FFTW here either, so the difference to the author's page is not caused by
    this fork's code.
*   Upstream's latest release is 11.6e (2023-12-21). This fork's headers still
    say version 11.4xv and the README says changes from 11.5 are merged, so the
    fork is at best at 11.5.

Compiler comparison: GCC and Clang
-----------------------------------

The same code was also built and measured with Clang 21.1.8 (Ubuntu, OpenMP
runtime `libomp`, OpenMP 5.1) on the same machine, with the same options,
Release build. The unit tests pass with Clang (22 of 22 test cases). FFTW is
the same library in both runs; PocketFFT and KissFFT are recompiled by the
compiler under test. Raw data: `docs/benchmark-data/2026-11-23/clang-*.csv`,
and `gcc-again-t1.csv` for a GCC re-run of the complex power-of-two sizes made
in the same session as the Clang runs.

**Summary:** the compiler does not change the picture. For the complex FFT on
powers of two OTFFT is within a few percent of its GCC speed for N up to
16,384 and 9 to 15 percent slower above that; FFTW (an identical binary) also
measured 7 to 9 percent lower from N=262,144 up in the Clang session, so part of
that is the state of the machine. The real FFT and the DCT are 4 to 12 percent
slower with Clang. The one clear difference is Bluestein, which is 15 to 43
percent slower with Clang.

**Complex FFT, powers of two, one thread, MFLOPS.** "GCC" is the re-run made
next to the Clang run. The FFTW columns show how much the machine itself moved
between the main GCC run and the Clang session (same binary).

| N | OTFFT GCC | OTFFT Clang | Clang vs GCC | FFTW, main GCC run | FFTW, Clang session |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 21,475 | 20,240 | -6% | 20,846 | 20,290 |
| 64 | 29,555 | 27,764 | -6% | 30,741 | 30,818 |
| 256 | 29,778 | 28,866 | -3% | 37,775 | 37,764 |
| 1,024 | 29,053 | 29,003 | -0% | 35,771 | 37,117 |
| 4,096 | 24,434 | 24,913 | +2% | 27,356 | 27,287 |
| 16,384 | 18,628 | 18,450 | -1% | 19,744 | 22,086 |
| 65,536 | 7,240 | 6,402 | -12% | 21,515 | 21,910 |
| 262,144 | 8,436 | 7,654 | -9% | 16,624 | 15,501 |
| 1,048,576 | 5,608 | 4,872 | -13% | 8,176 | 7,474 |
| 4,194,304 | 5,980 | 5,074 | -15% | 4,093 | 3,704 |

**Other transforms, one thread, MFLOPS** (GCC from the main run above):

| Transform | N | OTFFT GCC | OTFFT Clang | Clang vs GCC |
| --- | ---: | ---: | ---: | ---: |
| real | 256 | 18,483 | 16,654 | -10% |
| real | 4,096 | 20,849 | 18,458 | -11% |
| real | 65,536 | 15,406 | 13,951 | -9% |
| real | 1,048,576 | 5,580 | 5,089 | -9% |
| dct | 256 | 11,303 | 10,227 | -10% |
| dct | 4,096 | 13,528 | 12,146 | -10% |
| dct | 65,536 | 12,193 | 10,787 | -12% |
| dct | 1,048,576 | 3,784 | 3,573 | -6% |
| complex, smooth | 480 | 10,676 | 10,949 | +3% |
| complex, smooth | 1,000 | 8,036 | 7,695 | -4% |
| complex, smooth | 10,080 | 2,271 | 2,243 | -1% |
| complex, smooth | 100,000 | 6,452 | 6,147 | -5% |

**Bluestein, one thread, MFLOPS:**

| N | `createBluesteinFFT` GCC | `createBluesteinFFT` Clang | Clang vs GCC |
| ---: | ---: | ---: | ---: |
| 480 | 2,389 | 1,522 | -36% |
| 1,000 | 2,630 | 1,580 | -40% |
| 3,000 | 1,471 | 976 | -34% |
| 10,080 | 1,461 | 1,044 | -29% |
| 100,000 | 815 | 667 | -18% |
| 1,000,000 | 739 | 628 | -15% |
| 1,009 | 2,775 | 1,580 | -43% |
| 10,007 | 1,462 | 1,086 | -26% |
| 100,003 | 778 | 660 | -15% |

**12 threads, complex FFT, powers of two, MFLOPS** (GCC from the main run):

| N | OTFFT GCC | OTFFT Clang | FFTW, main GCC run | FFTW, Clang session |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 22,663 | 23,651 | 13,697 | 14,477 |
| 16,384 | 7,256 | 7,855 | 31,849 | 31,318 |
| 65,536 | 14,427 | 14,114 | 41,944 | 46,655 |
| 262,144 | 16,901 | 18,582 | 47,953 | 55,844 |
| 1,048,576 | 20,588 | 19,965 | 24,250 | 25,086 |
| 4,194,304 | 10,244 | 11,742 | 17,154 | 16,488 |

**Upstream-style round trip** (OTFFT time as a percentage of FFTW
`FFTW_MEASURE`, one thread, median over 2^1 to 2^21): upstream 80% (its own
Clang 8 on a Mac), this repository with GCC 15: 111%, with Clang 21:
112%. So switching to Clang does **not** bring this fork's numbers
towards upstream's, which weakens the explanation "upstream used another
compiler". Remaining candidates are the CPU (Zen 2 against a 2016 Intel core),
the FFTW version and the fork's code and algorithm table
(`docs/upstream-comparison.md`).

Caveats: differences below about 10 percent are within the noise of this
shared machine; PocketFFT and KissFFT varied by up to 38 percent between
sessions at small sizes, so they are not compared here. The Clang binary was
built before `-fsignaling-nans` was restricted to GCC (PR #22); Clang ignored
that flag, so the generated code is the same. The problems of #16 to #19 occur
with both compilers.

Same-session comparison with upstream 11.6e
-------------------------------------------

Upstream OTFFT 11.6e was downloaded, built as released (its own generated
tables, `-O3 -march=native -fopenmp`, GCC 15.2) and run through the same harness
(`OTFFT_UPSTREAM_DIR`, see "Reproduce"), on the same machine and in the same
session as this fork and FFTW. All figures are the fork's MFLOPS divided by
upstream's, so below 1.0 the fork is slower. Upstream's `ffttune` was not run.

| Transform | 1 thread, N up to 16,384 | 12 threads, N up to 16,384 |
| --- | ---: | ---: |
| Complex | 1.00 | 0.95 |
| Real | 0.98 | 1.00 |
| DCT | 0.95 | 0.96 |
| Round trip | 0.97 | 0.95 |

*Geometric mean over 10 sizes (14 for the round trip); a single run each. At
1 thread the range over the sizes is 0.92 to 1.12, so this fork is equal to
upstream for small and medium lengths, give or take a few percent. At 12
threads complex N=4,096 is 0.64.*

For N >= 65,536 the large sizes were run three times each (`large-*` files in
the data directory; the min and max over the three runs are close, so these
differences are reproducible, not noise). Median ratios:

| N | Complex 1 thr | Complex 12 thr | Real 1 thr | Real 12 thr | DCT 1 thr | DCT 12 thr |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 0.53 (0.48 to 2.68) | 1.10 | 0.94 | 0.96 | 0.89 | 1.00 |
| 262,144 | 0.63 | 1.36 | 1.19 | 0.79 | 1.08 | 0.84 |
| 1,048,576 | 0.92 | 1.44 | 0.78 | 0.49 | 0.82 | 0.59 |
| 4,194,304 | 1.09 | 1.25 | 0.98 | 0.60 | 1.03 | 0.72 |

| N | Round trip 1 thr | Round trip 12 thr |
| ---: | ---: | ---: |
| 131,072 | 1.23 | 0.72 |
| 262,144 | 0.61 | 1.63 |
| 524,288 | 0.62 | 0.36 |
| 1,048,576 | 0.90 | 1.63 |
| 2,097,152 | 1.00 | 0.46 |
| 4,194,304 | 1.07 | 1.32 |
| 8,388,608 | 0.96 | 0.55 |
| 16,777,216 | 0.91 | 1.03 |

**What this shows**

*   **Speed relative to FFTW.** Upstream 11.6e is not faster than FFTW here
    either. Its geometric mean over the sizes of the main run, as a fraction of
    FFTW-measure's speed with one thread, is 0.55 for complex, 0.59 for real and
    1.34 for DCT; the fork's is 0.52, 0.58 and 1.37. The author's claim
    therefore depends on the machine, compiler or FFTW version, not on code
    that this fork lost. The compiler alone is not the reason (the fork gives
    about the same with Clang, see "Compiler comparison").
*   **No regression for small and medium N.** Up to N=16,384 the fork and
    upstream agree within a few percent, apart from complex N=4,096 with 12
    threads (0.64) and DCT, which is about 5% behind.
*   **Single-thread large complex sizes.** The fork is slower at N=65,536
    (0.53) and from 262,144 to 524,288 (0.6). These are sizes where upstream's
    per-size algorithm table differs (15 of 24 entries, see
    `docs/upstream-comparison.md`) and where the `exp` kernels start (2^18).
    Not verified which of the two is responsible.
*   **12 threads.** The fork is faster for complex at all large sizes (1.1 to
    1.4) but 20% to 50% slower for real and DCT from 262,144 up. The round trip
    swings between 0.36 and 1.63 with the exponent, alternating odd and even.
    This points to different algorithms or thresholds being chosen per size in
    the two versions (upstream's OpenMP changes, roadmap R1.10), which also
    bears on [#19](https://github.com/DEWETRON/otfft/issues/19).
*   **Conclusion.** The fork is not generally behind upstream, but it loses
    clearly at some large sizes. Roadmap R1.8 to R1.10 are the right fix and
    now have a measured target: complex N=65,536 to 524,288 at one thread
    and real and DCT at 12 threads from N=262,144.

Caveats of this comparison: one machine, GCC only for upstream, upstream's tuner not run;
the main run is a single run, so single differences below about 10% at N <=
16,384 are not meaningful. FFTW-measure ran with the same 10 s planning limit
as before.

Intel Core Ultra 7 268V, Windows, MSVC
--------------------------------------

The same harness run on a Windows 11 notebook on 2026-11-24, to see how the
picture changes on another CPU and compiler. **FFTW was not available**, so
only OTFFT, PocketFFT and KissFFT are compared, and only complex transforms
(the default `--kinds`) were run. The prime length N=100,003 is not in the
default size set, so it is missing too.

| Item | Value |
| --- | --- |
| CPU | Intel Core Ultra 7 268V, 8 cores / 8 threads (no Hyper-Threading) |
| OS | Windows 11 Enterprise, notebook, default power plan, not isolated |
| Compiler | MSVC (Visual Studio 2022 Build Tools), `Release` (`/O2`) |
| Threads | 1 and 8 (the Ryzen runs used 12) |

The log says `optimised build: NO` and `cpu: unknown`. Both are false:
`bench.cpp` tests `__OPTIMIZE__`, which MSVC does not define, and reads the CPU
name only on Linux. The build was a Release build.

**Complex FFT, powers of two, one thread, MFLOPS**

| N | OTFFT | PocketFFT | KissFFT |
| ---: | ---: | ---: | ---: |
| 16 | **29,196** | 576 | 4,063 |
| 64 | **41,554** | 2,510 | 6,783 |
| 256 | **42,646** | 5,950 | 8,536 |
| 1,024 | **44,476** | 7,049 | 9,391 |
| 4,096 | **25,566** | 9,123 | 9,406 |
| 16,384 | **24,249** | 9,718 | 9,171 |
| 65,536 | **13,639** | 4,717 | 6,346 |
| 262,144 | **9,880** | 5,456 | 6,435 |
| 1,048,576 | **6,280** | 5,012 | 2,801 |
| 4,194,304 | **5,362** | 5,249 | 1,910 |

**Complex FFT, smooth and prime lengths, one thread, MFLOPS**

| N | OTFFT | PocketFFT | KissFFT | OTFFT-bluestein |
| ---: | ---: | ---: | ---: | ---: |
| 480 | **19,809** | 7,139 | 8,613 | 2,239 |
| 1,000 | **14,016** | 7,875 | 8,870 | 2,390 |
| 3,000 | **13,910** | 8,612 | 8,647 | 1,472 |
| 10,080 | 5,303 | **8,906** | 4,900 | 1,296 |
| 100,000 | **11,402** | 5,788 | 8,536 | 1,069 |
| 1,000,000 | **9,976** | 6,112 | 3,126 | 1,116 |
| 1,009 (prime) | 39 | 1,186 | 69 | **2,406** |
| 10,007 (prime) | 5 | 1,210 | 9 | **1,349** |

**OTFFT, powers of two, MFLOPS: backends at one thread, and 8 threads**

| N | SSE2 (1 thread) | AVX (1 thread) | AVX2 (1 thread) | auto, 1 thread | auto, 8 threads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 21,670 | 26,795 | 27,574 | 29,196 | 28,499 |
| 1,024 | 23,342 | 38,489 | 42,497 | 44,476 | 43,557 |
| 16,384 | 16,878 | 23,649 | 19,151 | 24,249 | 11,914 |
| 65,536 | 9,820 | 10,976 | 11,807 | 13,639 | 15,995 |
| 262,144 | 8,934 | 12,197 | 11,500 | 9,880 | 22,715 |
| 1,048,576 | 7,946 | 6,516 | 6,208 | 6,280 | 15,302 |
| 4,194,304 | 5,371 | 5,213 | 5,162 | 5,362 | 17,286 |

What this shows:

*   OTFFT is the fastest of the three at every power of two. Its lead over
    PocketFFT falls from about 6x at N=1,024 to 1.0x at N=4,194,304, where
    both are limited by memory.
*   N=10,080 is again OTFFT's weak smooth length: PocketFFT is 1.7x faster.
*   The prime-length problem is the same as on the Ryzen: `createComplexFFT`
    gives 39 and 5 MFLOPS, `createBluesteinFFT` is 2,406 and 1,349 and beats
    PocketFFT at N=1,009 (2.0x) and N=10,007 (1.1x).
*   AVX and AVX2 are 1.5x to 1.9x faster than SSE2 up to N=1,024. Above
    N=1,048,576 and up SSE2 is as fast or faster.
*   8 threads are faster from N=65,536 (1.2x) up to 3.2x at N=4,194,304 and
    2.4x at N=1,048,576, but slower at N=16,384 (0.5x), as on the Ryzen.
*   Absolute numbers cannot be compared with the Ryzen tables: another CPU,
    another compiler, and a notebook with power management. There are no FFTW
    numbers to set against.

Raw results and logs: `docs/benchmark-data/2026-11-24-intel/` (`bench-t1.*`,
`bench-t8.*`). Reproduce as in "Reproduce" below, with `--threads 8`; the
Windows build uses the Visual Studio generator, so the executable is
`build/Release/otfft_bench.exe`.

Algorithm selection baseline (ffttune), Intel Core Ultra 7 268V
---------------------------------------------------------------

Baseline for roadmap item R1.8 (re-tune the per-size algorithm table in
`src/otfft_algo.cpp`). Both OTFFT versions pick one algorithm per power-of-two
length: upstream from a table generated by its `ffttune` tool, the fork from a
fixed table. This section records which algorithm is fastest on the Intel
notebook above and how much the fork's table loses. **Nothing was changed in
the library.**

Setup: one thread, AVX2 code compiled with MSVC `/O2`, same machine and state
as the section above. Two measurements:

*   **Upstream `ffttune` 11.6e**, built as released (`/O2 /arch:AVX2`) and run
    with no arguments. It times upstream's own classes (mean of up to 16 runs
    of a forward plus an inverse transform, outliers dropped).
*   **The fork's own classes**, timed by a small harness
    (`forktune.cpp`, kept with the data). It calls the `instance()` factories
    of the fork's AVX2 backend directly, runs a forward plus an inverse
    transform in 9 batches of about 20 ms, and takes the median. Two complete
    passes.

Algorithms: DIF4, DIT4, DIF8, DIT8, DIF16, DIT16 (AVX decimation-in-frequency
or -in-time with radix 4, 8 or 16) and SixStep.

**Fastest algorithm per length**

| log2 N | Upstream shipped | Fork now | Upstream classes, tuned here | Fork classes, pass 1 | Fork classes, pass 2 | Fork now vs fastest, pass 1 | Pass 2 |
| ---: | --- | --- | --- | --- | --- | ---: | ---: |
| 11 | DIF16 | DIF16 | DIF16 | DIF16 | DIF16 | 1.00x | 1.00x |
| 12 | DIF8 | DIT8 | DIF8 | DIF8 | DIF8 | 1.11x | 1.15x |
| 13 | DIF8 | DIF8 | DIF8 | DIF8 | DIF8 | 1.00x | 1.00x |
| 14 | DIT8 | DIT8 | DIF8 | DIF8 | DIF8 | 1.09x | 1.10x |
| 15 | DIF8 | DIT8 | DIF8 | DIF8 | DIF8 | 1.47x | 1.50x |
| 16 | DIF16 | SixStep | DIF8 | DIF8 | DIF8 | 1.32x | 1.33x |
| 17 | DIF16 | DIT8 | DIF8 | DIF8 | DIF8 | 1.65x | 1.61x |
| 18 | DIF16 | SixStep | DIF8 | DIF8 | DIF8 | 1.29x | 1.22x |
| 19 | DIF16 | SixStep | DIF8 | DIF16 | DIF16 | 1.25x | 1.26x |
| 20 | DIF16 | SixStep | DIF8 | DIF8 | DIF8 | 1.33x | 1.71x |
| 21 | SixStep | SixStep | DIF8 | DIF8 | DIF8 | 1.26x | 1.31x |
| 22 | DIF16 | SixStep | DIF8 | DIF8 | DIF8 | 1.85x | 1.68x |
| 23 | DIF16 | SixStep | DIF8 | DIF8 | DIF8 | 1.37x | 1.17x |
| 24 | DIF16 | SixStep | DIF8 | DIF8 | DIF8 | 1.80x | 1.77x |

"Fork now vs fastest" is the time of the algorithm the fork uses today divided
by the time of the fastest one in the same pass (1.00x means the fork already
uses the fastest). Lengths up to 2^10 are left out: the algorithms are within
the measurement noise there (and the two passes disagree).

What this shows:

*   On this machine DIF8 is the fastest algorithm for most lengths from 2^12
    to 2^24, with both OTFFT versions' classes. The exceptions are 2^11 (DIF16)
    and 2^19, where DIF16 and DIF8 are about 10% apart in both passes.
*   The fork's table loses 1.1x to 1.8x from 2^14 up, most at 2^15 and 2^17
    (DIT8) and for SixStep at 2^22 and 2^24. Upstream's shipped table (mostly
    DIF16) is also not the fastest here.
*   The tuning depends on the CPU, compiler and thread count. These numbers
    cover AVX2, MSVC and one thread on one Intel CPU only. They are a baseline
    to compare with, not a table to copy: the AMD Ryzen and GCC have not been
    measured, and the 2^20 and 2^22 to 2^23 ratios move by up to 0.5x between
    passes.

Raw results: `docs/benchmark-data/2026-11-24-intel/tune/`
(`upstream-ffttune.log`, `fork-avx2-run1.csv`, `fork-avx2-run2.csv` with the
time in microseconds of one forward plus one inverse transform per algorithm,
and the harness `forktune.cpp`). The harness is not part of the build; it was
compiled with
`cl /MD /openmp /O2 /std:c++17 /arch:AVX2 /DUSE_AVX2 /DOTFFT_NAMESPACE=OTFFT_AVX2 /Iinc /Isrc forktune.cpp`
and linked with `otfft_avx2.lib` and `otfft_common.lib` from a Release build.

Algorithm selection baseline (ffttune), AMD Ryzen 9 9950X
---------------------------------------------------------

Second baseline for roadmap item R1.8, measured with the fork's `ffttune` tool
(`-DOTFFT_BUILD_TOOLS=ON`, decision D12) on 2026-11-24. The question was
whether the Intel result above (DIF8 fastest from 2^12 up, fork's table 1.1x to
1.8x slower from 2^14) also holds on an AMD CPU with GCC. **Nothing was changed
in the library.**

Note that this is a different machine from the Ryzen 9 3900X in "Setup":

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads, L2 1 MiB per core, L3 2 x 32 MiB |
| OS | Linux 7.0.0-44-generic, driver `amd-pstate-epp`, governor `powersave` (not changed: `sudo` needs a password) |
| Compiler | GCC 15.2.0, Release |
| FFTW | 3.3.10 from the Ubuntu package (`libfftw3-dev`, SSE2/AVX), found by CMake |
| Machine state | Desktop in use: an Oxygen process was busy at about 65% of one core (load average about 1.4), plus VS Code. Not closed. Threads not pinned. |

Runs (`build/Release/ffttune --backend B --threads T --out ...`, medians of 9
batches of about 20 ms, forward plus inverse transform): AVX2 with one thread
three times (r1 to r3), AVX with one thread once, AVX2 with 12 threads once.
No result was marked wrong (`-`) in any run.

**Fastest algorithm per length, AVX2, one thread**

| log2 N | Fastest r1 | Fastest r2 | Fastest r3 | Fork now | Fork now vs fastest (r1, r2, r3) | Within 5% of the fastest (r1) |
| ---: | --- | --- | --- | --- | --- | --- |
| 11 | DIF16 | DIF16 | DIT8 | DIF16 | 1.00x, 1.00x, 1.00x | DIF4, DIT4, DIT8 |
| 12 | DIT4 | DIT4 | DIT4 | DIT8 | 1.01x, 1.03x, 1.05x | DIF4, DIT8 |
| 13 | DIF4 | DIF4 | DIF4 | DIF8 | 1.06x, 1.07x, 1.09x | |
| 14 | DIF4 | DIF4 | DIF4 | DIT8 | 1.23x, 1.21x, 1.19x | |
| 15 | DIF8 | DIF8 | DIF8 | DIT8 | 1.07x, 1.07x, 1.08x | DIF4 |
| 16 | DIF4 | DIF4 | DIF4 | SixStep | 3.08x, 3.09x, 3.09x | (r2: DIF8) |
| 17 | DIF4 | DIF4 | DIF4 | DIT8 | 1.12x, 1.14x, 1.16x | |
| 18 | DIF4 | DIF4 | DIF4 | SixStep | 3.28x, 3.39x, 3.32x | |
| 19 | DIF4 | DIF4 | DIF4 | SixStep | 3.05x, 3.11x, 3.03x | |
| 20 | DIF8 | DIF8 | DIF8 | SixStep | 1.96x, 2.02x, 2.04x | |
| 21 | DIF8 | DIF8 | DIF8 | SixStep | 1.46x, 1.48x, 1.49x | DIF16 |
| 22 | DIT16 | DIT16 | DIF16 | SixStep | 1.60x, 1.60x, 1.61x | DIF16 |
| 23 | DIT16 | DIT16 | DIT16 | SixStep | 1.52x, 1.47x, 1.47x | DIF16 |
| 24 | DIT16 | DIF16 | DIF16 | SixStep | 1.66x, 1.61x, 1.64x | DIF16 |

"Fork now" is the pick in `src/otfft_algo.cpp`; the ratio is its time divided
by the fastest algorithm's time in the same run. Lengths up to 2^10 are left
out (all within noise, as on the Intel machine).

What this shows:

*   **The result is stable between runs.** The fork's slowdown differs by at
    most 0.11x between r1 and r3 at every length. The fastest algorithm changes
    between runs only where two are within 5% of each other: 2^11 (DIF16 or
    DIT8), 2^22 and 2^24 (DIT16 or DIF16). 2^12 (DIT4, DIF4 and DIT8 within 5%)
    and 2^15 (DIF8 and DIF4 within 5%) are ties too.
*   **The fork's table is clearly wrong for one thread at the large sizes.**
    SixStep is 3.0x to 3.4x slower than the fastest at 2^16, 2^18 and 2^19, 2.0x
    at 2^20, and 1.5x to 1.7x at 2^21 to 2^24. This agrees with the benchmark
    below: OTFFT falls from 36,096 MFLOPS at N=16,384 to 11,091 at N=65,536
    while FFTW stays at 38,039. Below 2^16 the loss is small (at most 1.2x,
    and 1.0x at 2^11).
*   **Different from the Intel result.** There DIF8 was fastest almost everywhere
    from 2^12 up. Here DIF4 is fastest at 2^13, 2^14 and 2^16 to 2^19, DIF8
    only at 2^15, 2^20 and 2^21, and DIT16 or DIF16 (about equal) from 2^22. At
    2^22 to 2^24 DIT16 or DIF16 is 1.2x to 1.3x faster than DIF8 (r1). What both machines agree on is
    that SixStep is not the best one-thread choice for 2^16 and up, and that
    DIT8 at 2^15 and 2^17 is not either.
*   **AVX (r1).** Same picture as AVX2, with ties moving (2^17 DIF8 or DIT8
    within 5%, 2^21 and 2^22 DIF16 first). The fork's slowdown is the same
    within 0.1x.
*   **12 threads (r1 only).** The picture reverses. SixStep is fastest at
    2^13 to 2^18, 2^20, 2^22 and 2^24, where the fork's pick is SixStep
    already (1.00x) at 2^16, 2^18, 2^20, 2^22 and 2^24 and 1.25x to 1.38x slower
    at 2^13, 2^14, 2^15 and 2^17 (DIF8 and DIT8). At 2^19, 2^21 and 2^23 SixStep
    is not fastest: the fork's pick is 4.1x, 3.2x and 2.0x slower than DIF16,
    DIT8 and DIT16. One run only, so the odd 2^19 and 2^21 results are not
    established. The threads were not pinned and 12 of 32 logical CPUs were
    used.

**Do the data suggest changing the table?** Yes for one thread: a
Ryzen/GCC table with DIF4 at 2^13, 2^14, 2^16 to 2^19, DIF8 at 2^15, 2^20 and
2^21 and DIT16 from 2^22 would remove the 1.5x to 3.4x loss above 2^16 on this
machine. But the table has no thread dimension, and the 12-thread run shows
that the best choice for 12 threads (SixStep) is the worst one for one
thread. So a change should wait for repeated multi-thread runs (the odd
2^19 and 2^21 results first), and for a run on the Intel machine with GCC or Clang
to separate the CPU from the compiler. That is a separate pull request with
a test (AGENTS.md rule 1). This pull request proposes no table.

**Complex FFT on this machine (benchmark, GCC, MFLOPS, powers of two)**

The earlier tables are from the 3900X, so these are new numbers.

| N | OTFFT 1 thread | FFTW-measure 1 thread | PocketFFT 1 thread | OTFFT 12 threads | FFTW-measure 12 threads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 55,485 | 69,968 | 11,854 | 56,064 | 11,362 |
| 16,384 | 36,096 | 43,903 | 14,153 | 20,729 | 76,803 |
| 65,536 | 11,091 | 38,039 | 10,799 | 48,060 | 138,281 |
| 262,144 | 11,883 | 35,941 | 10,639 | 51,858 | 113,293 |
| 1,048,576 | 10,998 | 17,949 | 8,557 | 61,105 | 140,947 |
| 4,194,304 | 8,053 | 14,879 | 5,840 | 26,010 | 30,788 |

OTFFT is 1.0x to 1.3x behind FFTW with one thread up to N=16,384 and 1.6x to
3.4x behind from N=65,536. The OTFFT column is the library's default pick (the
table in question), so part of that gap is the SixStep problem above. All sizes,
the backends and the plan times are in the CSV files.

Limits: one CPU, one compiler (GCC 15.2.0), no `performance` governor
(`powersave` with `amd-pstate-epp`, so clock speed varied with load; the CPU
reports a 5.76 GHz maximum), another program running during all runs, threads
not pinned, and three runs for the main case but one each for AVX and 12
threads. Run-to-run differences under 5% are noise. FFTW here is the
distribution package (SSE2/AVX), not the AVX2 build used for the 3900X
numbers, so the FFTW columns are not comparable with the earlier ones.

Raw results: `docs/benchmark-data/2026-11-24-ryzen-linux/` (`bench-t1.*`,
`bench-t12.*` from `otfft_bench --kinds complex`; the `.log` files are from a
second run with the same command, whose results agree with the CSVs within 7%
at the sizes compared) and
`docs/benchmark-data/2026-11-24-ryzen-linux/tune/`
(`tune-avx2-t1-r1..r3`, `tune-avx-t1-r1`, `tune-avx2-t12-r1`, each as `.log`
with the full table and `.txt` with the suggested `case` lines).

Algorithm table change: effect on the Intel notebook
----------------------------------------------------

Roadmap R1.8, decision D14. From the two tuning baselines above, `src/otfft_algo.cpp`
now has two algorithm tables: one for a configured OpenMP thread count of 1
(new, from the data of both machines) and one for more than one thread (the
old table, unchanged). This section measures the change on the Intel notebook
(MSVC, Release), `otfft_bench` of `master` before the change and of the
branch after it, built in the same session and run alternately: complex
transforms, powers of two, `OTFFT` (default algorithm choice), MFLOPS.

| N | 1 thread, before (r1, r2) | 1 thread, after (r1, r2) | Change | 8 threads before | 8 threads after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 22,744 / 26,610 | 27,379 / 24,862 | 1.06x | 22,231 | 24,453 |
| 16,384 | 19,231 / 14,684 | 23,524 / 23,110 | 1.38x | 7,086 | 6,858 |
| 65,536 | 13,092 / 12,234 | 14,541 / 14,697 | 1.15x | 13,286 | 8,786 |
| 262,144 | 10,070 / 9,491 | 11,945 / 13,317 | 1.29x | 18,028 | 18,762 |
| 1,048,576 | 5,511 / 6,576 | 11,593 / 10,964 | 1.87x | 17,661 | 14,829 |
| 4,194,304 | 4,776 / 4,955 | 8,342 / 8,700 | 1.75x | 16,354 | 16,523 |

The change column is the mean of the two "after" runs over the mean of the two
"before" runs.

*   With one thread the new table is faster at every size from 16,384
    (1.15x to 1.87x), in line with the tuning data. At N=4,096 the pick did not
    change (DIT8 in both tables), so the 1.06x there is run noise.
*   With 8 threads the table did not change, so the differences in the last two
    columns (up to 0.66x at N=65,536) are run-to-run noise of one run each: the
    same code runs in both. This is also why the 8-thread table was not
    re-tuned here.
*   One CPU, MSVC, two runs per cell at one thread. On the Ryzen (GCC) the
    tuning data predict gains of 1.5x to 3.3x with one thread above 2^15;
    that run was not repeated with the new library.

Raw results: `docs/benchmark-data/2026-11-24-intel/table-change/`
(`old-*` and `new-*`).

Algorithm table change: effect on the Ryzen 9 9950X
---------------------------------------------------

The same change as above (roadmap R1.8, decision D14), measured on the AMD
Ryzen 9 9950X with GCC. The ffttune data of that machine predicted 1.5x to 3.3x
with one thread from 2^16 and no change with several threads.

Before: `master` at `bbb6c0c` (first parent of the merge of PR #31). After:
`master` at `b548629`. Both were built Release with GCC 15.2.0 in separate
directories, and run alternately (old, new, old, new, ...) as
`otfft_bench --kinds complex --sizes pow2 --min-n 4096 --lib OTFFT`, MFLOPS of
the default `OTFFT` choice. Machine: Linux 7.0.0-44, `amd-pstate-epp` with
governor `powersave` (not changed: `sudo` needs a password), other load a
`uniThreadPool` process (Oxygen) at about 60% of one core and VS Code, load
average about 1.5 to 1.9, threads not pinned. `ctest` on the new build:
`otfft.test` and `ffttune.smoke` passed.

**One thread, three alternating runs (MFLOPS)**

| N | Before r1 / r2 / r3 | Before mean | After r1 / r2 / r3 | After mean | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 42,020 / 46,092 / 48,326 | 45,480 | 48,002 / 47,132 / 45,671 | 46,935 | 1.03x |
| 16,384 | 35,963 / 36,651 / 35,887 | 36,167 | 38,727 / 39,453 / 39,734 | 39,305 | 1.09x |
| 65,536 | 11,195 / 11,281 / 11,177 | 11,218 | 33,410 / 33,374 / 33,329 | 33,371 | **2.97x** |
| 262,144 | 12,250 / 12,294 / 11,911 | 12,152 | 35,120 / 35,769 / 35,723 | 35,538 | **2.92x** |
| 1,048,576 | 11,181 / 11,038 / 11,049 | 11,089 | 20,468 / 21,640 / 20,359 | 20,823 | **1.88x** |
| 4,194,304 | 7,870 / 7,872 / 8,060 | 7,934 | 12,592 / 12,743 / 12,933 | 12,756 | **1.61x** |

The change is the mean of the three "after" runs over the mean of the three
"before" runs. Spread between the three runs (highest minus lowest, over the
mean): at most 6% for N >= 16,384, except 14% for "before" at N=4,096.

**12 and 16 threads, three alternating runs each (mean MFLOPS)**

The table for more than one thread did not change, so any difference here is
noise. The first run of each is `old-t12.csv`, `new-t12.csv` (`-t16`); the
other two (`-r2`, `-r3`) were added after the first run showed large
differences.

| N | 12 thr. before | 12 thr. after | Change | Spread before / after | 16 thr. before | 16 thr. after | Change | Spread before / after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 44,974 | 47,231 | 1.05x | 11% / 8% | 46,695 | 47,625 | 1.02x | 10% / 5% |
| 16,384 | 20,418 | 18,276 | 0.90x | 16% / 20% | 16,700 | 17,837 | 1.07x | 8% / 25% |
| 65,536 | 44,150 | 45,325 | 1.03x | 7% / 8% | 40,975 | 42,964 | 1.05x | 4% / 8% |
| 262,144 | 56,363 | 51,237 | 0.91x | 14% / 5% | 56,523 | 55,504 | 0.98x | 5% / 4% |
| 1,048,576 | 55,922 | 54,237 | 0.97x | 6% / 10% | 58,580 | 59,587 | 1.02x | 9% / 22% |
| 4,194,304 | 23,861 | 24,610 | 1.03x | 5% / 4% | 24,705 | 25,311 | 1.02x | 2% / 2% |

**One-thread comparison with FFTW** (one run of the full benchmark with the new
build, `new-fftw-t1.csv`; FFTW 3.3.10 Ubuntu package, `FFTW_MEASURE`; MFLOPS)

| N | OTFFT after | FFTW | OTFFT vs FFTW | Before the change (earlier section) |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 33,745 | 37,324 | 0.90x | 0.29x |
| 262,144 | 35,239 | 34,505 | 1.02x | 0.33x |
| 1,048,576 | 21,509 | 17,756 | 1.21x | 0.61x |
| 4,194,304 | 12,986 | 15,049 | 0.86x | 0.54x |

"Before" is the OTFFT over FFTW ratio of the Ryzen benchmark of the tuning
section above (a different run on the same machine).

What this shows:

*   **The predicted gain appeared.** With one thread the new table is 2.97x
    (N=65,536), 2.92x (262,144), 1.88x (1,048,576) and 1.61x (4,194,304)
    faster. The ffttune data predicted 3.1x at 2^16, 3.3x at 2^18, 2.0x at 2^20
    and 1.6x at 2^22, so all four agree within 12%. The three runs agree
    within 6%. At N=16,384 the gain is 1.09x (predicted 1.2x for the old DIT8
    pick); at N=4,096 the pick did not change and the 1.03x is noise (the
    "before" runs alone differ by 14%). No one-thread size got slower.
*   **Multi-thread results are mostly unchanged, with two means at the edge
    of the noise.** Most changes are within 0.97x to 1.07x. At 12 threads the
    means are 0.90x at N=16,384 and 0.91x at N=262,144. The run-to-run spread
    of one cell is 14% to 20% in those cells, so three runs do not separate
    this from noise, but we cannot exclude a small real loss either. The
    table for threads is the same code path, so no cause is known. At
    16 threads no cell is below 0.98x. Spreads in the threaded runs are
    larger than at one thread (up to 25%); the odd N=16,384 and
    N=1,048,576 cells ("after", 16 threads) each have one run far from the other
    two.
*   **Where the new table is still behind FFTW** (one thread): 1.11x at
    N=65,536 and 1.16x at N=4,194,304 (single run each; at 4,096 and 16,384
    FFTW leads by 1.06x and 1.11x). OTFFT is level or ahead at 262,144 and
    1,048,576. This closes most of the gap of 1.6x to 3.4x found before, but
    not all of it. With several threads FFTW was far ahead before (1.2x to 3.7x at
    N=16,384 to 4,194,304 in the 12-thread table of the tuning section above) and this change did not touch that;
    the threaded FFTW numbers were not repeated here.
*   **Limits.** One CPU, GCC 15.2.0, `powersave` governor, other load
    present, threads not pinned. Three runs per one-thread cell, three per
    threaded cell. The FFTW comparison is one run, and its OTFFT numbers differ by
    up to 10% from the means above (51,514 against 46,935 at N=4,096), which
    is the size of the noise at the small N. `--sizes pow2` covers every
    second power of two only (2^12, 2^14, ...), so 2^15, 2^17 and the odd
    exponents above were not measured with the library.

Raw results: `docs/benchmark-data/2026-11-24-ryzen-linux/table-change/`
(`old-*` and `new-*` CSV and log files, `new-fftw-t1.*`).

Radix-7 butterfly: effect on the Intel notebook
-----------------------------------------------

Roadmap R1.5 (first part), issue #18, decision D15. The mixed radix code had
butterflies for the factors 2, 3, 4, 5 and 8 only; a factor 7 went through the
generic O(k^2) loop. It now has a radix-7 butterfly (forward, inverse, and the
OpenMP versions) that uses the symmetry of the 7th roots of unity: 18
multiplications by a real constant instead of 36 complex multiplications.

Same-session comparison on the Intel notebook above (MSVC, Release, one
thread): `otfft_bench` of `master` before the change and of the branch after
it, built the same way and run alternately (two runs each); complex transforms,
MFLOPS.

| N | Factors | Before (r1, r2) | After (r1, r2) | Change | PocketFFT |
| ---: | --- | ---: | ---: | ---: | ---: |
| 480 | 2^5 3 5 | 17,181 / 16,915 | 18,450 / 18,255 | 1.08x | 6,325 |
| 1,000 | 2^3 5^3 | 13,874 / 13,941 | 13,074 / 13,222 | 0.95x | 6,073 |
| 3,000 | 2^3 3 5^3 | 13,197 / 12,730 | 12,030 / 12,646 | 0.95x | 5,519 |
| **10,080** | 2^5 3^2 5 **7** | 7,205 / 7,600 | **14,529 / 14,684** | **1.97x** | 8,330 |
| 100,000 | 2^5 5^5 | 9,611 / 10,168 | 9,624 / 9,771 | 0.98x | 5,177 |
| 1,000,000 | 2^6 5^6 | 8,467 / 9,498 | 8,733 / 9,152 | 1.00x | 5,684 |

Lengths without a factor 7 do not use the new code; their changes (0.95x to
1.08x) are run-to-run noise. The benchmark has only one length with a 7, so a
second, small timing program (`lens.cpp`, kept with the data) times
`createComplexFFT` for more lengths, before and after, one run each:

| N | Factors | Before | After | Change |
| ---: | --- | ---: | ---: | ---: |
| 7 | 7 | 2,852 | 11,874 | 4.2x |
| 49 | 7^2 | 2,779 | 15,375 | 5.5x |
| 2,401 | 7^4 | 3,137 | 15,911 | 5.1x |
| 117,649 | 7^6 (OpenMP) | 7,350 | 36,509 | 5.0x |
| 112 | 2^4 7 | 6,017 | 19,711 | 3.3x |
| 245 | 5 7^2 | 3,753 | 14,518 | 3.9x |
| 3,500 | 2^2 5^3 7 | 7,405 | 14,166 | 1.9x |
| 5,600 | 2^5 5^2 7 | 8,109 | 16,896 | 2.1x |
| 10,080 | 2^5 3^2 5 7 | 8,789 | 17,311 | 2.0x |
| 6,720 | 2^6 3 5 7 | 8,473 | 18,962 | 2.2x |

*   A factor 7 is now 1.9x to 5.5x faster. OTFFT is 1.75x faster than
    PocketFFT at N=10,080 (it was 0.9x).
*   Lengths with a 7 are now as fast per point as lengths with only 3 and 5
    (14,000 to 19,000 MFLOPS, powers of 7 up to 16,000). The remaining gap to
    FFTW (on the Ryzen, 2.8x to 3.5x for N = 480, 1,000, 3,000) comes from the
    radix 3 and 5 butterflies, which are still the plain form with complex
    multiplications; that is the next step of R1.5. FFTW is not available on
    this machine, so no FFTW numbers were measured here.
*   One CPU, one compiler, one thread; the second table is single runs.

Raw results: `docs/benchmark-data/2026-11-24-intel/radix7/`.

Radix-3 and radix-5 butterflies in the symmetric form: effect on the Intel notebook
--------------------------------------------------------------------------------

Roadmap R1.5 (second part), issue #18, decision D16. After the radix-7
butterfly (previous section), the radix-3 and radix-5 butterflies were the
remaining plain ones (4 and 16 complex multiplications per butterfly). They use
the symmetric form with real constants now (2 and 8 multiplications by a real
number), forward and inverse, with and without OpenMP.

Method: a small timing program (`lens.cpp`, kept with the data) times
`createComplexFFT` `fwd0` for each length (best of 7 timed loops of about 30 ms),
of the library before and after the change, one thread, MSVC Release, Intel
notebook as above. The two libraries were run alternately, three times each;
the table shows the median of the three, MFLOPS (5 N log2 N / time).

| N | Factors | Before | After | Change |
| ---: | --- | ---: | ---: | ---: |
| 3 | 3 | 4,716 | 4,294 | 0.91x |
| 27 | 3^3 | 11,299 | 12,105 | 1.07x |
| 2,187 | 3^7 | 14,622 | 15,768 | 1.08x |
| 59,049 | 3^10 (OpenMP) | 16,909 | 15,214 | 0.90x |
| 5 | 5 | 6,530 | 8,715 | 1.33x |
| 25 | 5^2 | 9,999 | 14,168 | 1.42x |
| 125 | 5^3 | 11,546 | 15,676 | 1.36x |
| 15,625 | 5^6 | 12,063 | 15,175 | 1.26x |
| 78,125 | 5^7 (OpenMP) | 19,665 | 24,715 | 1.26x |
| 15 | 3 5 | 9,020 | 10,090 | 1.12x |
| 75 | 3 5^2 | 11,122 | 14,703 | 1.32x |
| 375 | 3 5^3 | 11,803 | 15,466 | 1.31x |
| 3,375 | 3^3 5^3 | 12,415 | 14,539 | 1.17x |
| 480 | 2^5 3 5 | 18,729 | 20,775 | 1.11x |
| 1,000 | 2^3 5^3 | 14,709 | 17,940 | 1.22x |
| 3,000 | 2^3 3 5^3 | 14,391 | 17,311 | 1.20x |
| 12,000 | 2^5 3 5^3 | 14,488 | 16,437 | 1.13x |
| 100,000 | 2^5 5^5 | 22,816 | 28,324 | 1.24x |
| 10,080 | 2^5 3^2 5 7 | 16,286 | 17,879 | 1.10x |
| 1,024 | 2^10 | 47,039 | 48,495 | 1.03x |
| 4,096 | 2^12 | 27,367 | 26,446 | 0.97x |
| 65,536 | 2^16 | 21,921 | 20,559 | 0.94x |

*   Lengths with a factor 5 are 1.1x to 1.4x faster: 5^k alone 1.25x to
    1.42x, mixed lengths such as 75, 375 and 1,000 1.2x to 1.3x, N=100,000
    1.24x. Between the three runs the results of most lengths vary by 10% or less
    (27% for the old library at N=78,125 and 23% at N=100,000).
*   Powers of 3 alone show no measurable change (0.90x to 1.12x, up to 12%
    spread between runs). A likely reason is that the radix-3 pass was already
    cheap per point and the transform is limited by memory access, not by the
    multiplications; this was not investigated.
*   Lengths made of 2 only (the last three rows) do not use the new code; their
    changes (0.94x to 1.03x) show the size of the noise.
*   With the radix-7 change, N=10,080 is now at about 18,000 MFLOPS (it was 8,800
    two steps ago).
*   `otfft_bench` of the same two builds (complex, smooth and power-of-two
    lengths, one thread, three runs each) showed the same direction for lengths with
    a factor 5 (median gains of 1.2x to 1.4x for N = 480, 1,000, 100,000 and
    1,000,000), but its run-to-run spread was up to 2x to 3x on this machine
    (probably the hybrid CPU moving the thread between core types) and pure
    powers of 2, which do not use the changed code, came out between 0.84x and
    1.69x. So none of its numbers are quoted; the files are kept
    in `docs/benchmark-data/2026-11-24-intel/radix35/bench-noisy/`.
*   FFTW is not available for MSVC on this machine; the GCC part below has the
    FFTW numbers.

**GCC, with FFTW (WSL2).** The same two builds (old = `master` with the radix-7
butterfly, new = with the radix 3 and 5 butterflies) were also built with
GCC 15.2.0 in WSL2 (Ubuntu 26.04) on the same Intel notebook, against FFTW 3.3.10
(Ubuntu package, SSE2/AVX) and PocketFFT. `otfft_bench`, complex transforms,
one thread, pinned to one logical CPU with `taskset -c 2` (which core type that
is was not checked), `--lib OTFFT`: old and new alternately three times
(medians below; the three runs differ by at most 1.05x for nearly all lengths,
1.20x at the worst); FFTW and PocketFFT from one separate run of the new build.
MFLOPS; "new / FFTW" is the new OTFFT divided by FFTW-measure (below 1 means
FFTW is faster). `ctest` passes with this build, and it has no compiler warnings from the OTFFT sources.

| N | Factors | Before | After | Change | FFTW-measure | New / FFTW | PocketFFT |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 480 | 2^5 3 5 | 14,906 | 18,693 | 1.25x | 41,163 | 0.45 | 6,651 |
| 1,000 | 2^3 5^3 | 11,217 | 16,251 | 1.45x | 35,524 | 0.46 | 8,537 |
| 3,000 | 2^3 3 5^3 | 10,797 | 15,314 | 1.42x | 30,495 | 0.50 | 7,419 |
| 10,080 | 2^5 3^2 5 7 | 13,585 | 15,549 | 1.14x | 28,658 | 0.54 | 10,468 |
| 100,000 | 2^5 5^5 | 8,787 | 12,951 | 1.47x | 26,608 | 0.49 | 8,384 |
| 1,000,000 | 2^6 5^6 | 8,051 | 10,691 | 1.33x | 11,669 | 0.92 | 6,694 |
| 16 | 2^4 | 33,589 | 32,768 | 0.98x | 30,752 | 1.07 | 1,694 |
| 256 | 2^8 | 37,611 | 38,457 | 1.02x | 45,756 | 0.84 | 7,488 |
| 1,024 | 2^10 | 37,106 | 37,328 | 1.01x | 46,483 | 0.80 | 8,980 |
| 4,096 | 2^12 | 23,915 | 25,034 | 1.05x | 36,245 | 0.69 | 9,274 |
| 65,536 | 2^16 | 17,018 | 17,197 | 1.01x | 26,400 | 0.65 | 6,798 |
| 262,144 | 2^18 | 13,872 | 15,030 | 1.08x | 20,712 | 0.73 | 6,276 |
| 1,048,576 | 2^20 | 10,252 | 10,361 | 1.01x | 16,070 | 0.64 | 5,426 |
| 4,194,304 | 2^22 | 7,885 | 7,870 | 1.00x | 4,008 | 1.96 | 4,745 |

*   Under GCC the lengths with a factor 5 are 1.25x to 1.47x faster
    (N = 1,000: 1.45x, N = 3,000: 1.42x, N = 100,000: 1.47x, N = 10,080 with the 7:
    1.14x); powers of 2, which use none of the changed code, differ by 0.98x to
    1.08x. The Windows/MSVC timing program above gave 1.1x to 1.4x.
*   FFTW is still faster: OTFFT reaches 0.45 to 0.54 of FFTW's speed for N = 480,
    1,000, 3,000, 10,080 and 100,000 (FFTW is 1.9x to 2.2x faster), and 0.92 at
    N=1,000,000. Before this change the same lengths were 0.32 to 0.36 (2.8x
    to 3.2x slower) for N = 480, 1,000 and 3,000. N=10,080 was about 11x behind on
    the Ryzen before the radix-7 change; this machine was not measured against
    FFTW before it. Powers of 2: 0.80 to 0.86 of FFTW for N = 64 to 1,024
    (OTFFT is ahead at N=16), 0.65 to 0.73 for N = 4,096 to 262,144, 0.64 at
    1,048,576, and OTFFT is ahead at N=4,194,304 (1.96x; FFTW measured slower
    than OTFFT there, and sizes that large vary by up to 30% between runs).
*   Limits: WSL2 on a hybrid CPU, one logical CPU, one run of FFTW and
    PocketFFT; the machine had a load average of about 2.6 when the runs
    started (builds just before) and was otherwise idle. The numbers are
    not comparable with the Ryzen tables.

Raw results: `docs/benchmark-data/2026-11-24-intel/radix35/` (MSVC timing
program, `bench-noisy/`) and `docs/benchmark-data/2026-11-24-intel/radix35-wsl-gcc/`
(GCC and FFTW in WSL2, with `machine.txt`).

Thread break-even and the OpenMP thresholds
-------------------------------------------

Roadmap R1.6 (first part), issue #19, decision D17. The question: from which
length does a transform run faster with several threads than with one? Each
AVX algorithm class decides this with its own constant `OMP_THRESHOLD` (below
it: one thread). The constants were 2^11 (DIF8, DIT16), 2^12 (DIF16), 2^13 (DIF4,
DIT4, DIT8) and 2^14 (SixStep); they came from upstream and were never measured
for this fork.

Method: `ffttune --backend avx2 --threads T` times every algorithm for each
length with T threads; dividing by the time with one thread gives the ratio
below (below 1 = threads help). Two data sets: the AMD Ryzen 9 9950X (native
Linux, GCC 15.2, 12 threads, one run; 1-thread times are the mean of three runs,
`docs/benchmark-data/2026-11-24-ryzen-linux/tune/`) and the Intel Core Ultra 7
268V (WSL2, GCC 15.2, 2, 4 and 8 threads, two runs each,
`docs/benchmark-data/2026-11-24-intel/threads/`).

**Ratio time(12 threads) / time(1 thread), Ryzen 9950X, AVX2**

| log2 N | DIF4 | DIT4 | DIF8 | DIT8 | DIF16 | DIT16 | SixStep |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 1.02 | 1.01 | 3.16 | 1.05 | 1.02 | 2.44 | 1.01 |
| 12 | 0.99 | 1.00 | 2.09 | 0.99 | 1.27 | 1.00 | 1.00 |
| 13 | 2.22 | 2.10 | 1.76 | 1.58 | 1.15 | 1.04 | 1.20 |
| 14 | 2.17 | 1.83 | 1.58 | 1.31 | 0.97 | 0.86 | 0.49 |
| 15 | 2.12 | 1.48 | 1.12 | 1.10 | 0.77 | 0.72 | 0.76 |
| 16 | 1.40 | 1.37 | 1.00 | 0.97 | 0.41 | 0.46 | 0.17 |
| 17 | 1.64 | 0.99 | 0.96 | 0.91 | 0.50 | 0.47 | 0.35 |
| 18 | 0.96 | 0.84 | 0.89 | 0.89 | 0.57 | 0.56 | 0.14 |
| 19 | 1.82 | 1.33 | 1.08 | 0.91 | 0.57 | 0.53 | 1.09 |
| 20 | 0.70 | 0.64 | 0.64 | 0.61 | 0.37 | 0.36 | 0.11 |
| 21 | 0.58 | 0.54 | 0.43 | 0.38 | 0.44 | 0.43 | 0.93 |
| 22 | 0.70 | 0.68 | 0.64 | 0.61 | 0.64 | 0.63 | 0.26 |
| old threshold | 2^13 | 2^13 | 2^11 | 2^13 | 2^12 | 2^11 | 2^14 |
| **new threshold** | **2^18** | **2^16** | **2^16** | **2^16** | **2^15** | **2^14** | **2^14** |

*   Threading starts far too early for DIF4, DIT4, DIF8 and DIT8: they are 1.1x
    to 3.2x slower with 12 threads up to 2^17 and only clearly faster from 2^18 to
    2^20. DIF16 and DIT16 break even at 2^15 and 2^14. SixStep scales from 2^14
    (but see the odd powers of 2 below).
*   On the Intel notebook with 2 threads the break-even sizes are lower (about
    2^14 for DIF4, DIT4, DIF16, 2^13 for DIT16, 2^15 for DIF8 and DIT8); with
    4 and 8 threads they are between 2^12 and 2^16 where the run was not
    disturbed (see "Limits").

**Best threaded time / best serial time (any algorithm), per length**

This is what a user gets: with threads, is the transform faster than with one
thread at all? (below 1 = yes)

| log2 N | Intel, 2 threads | 4 threads | 8 threads | Ryzen, 12 threads |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 0.90 | 0.77 | 0.93 | 1.00 |
| 13 | 1.01 | 0.72 | 2.06 | 1.48 |
| 14 | 0.92 | 0.72 | 0.74 | 1.21 |
| 15 | 0.75 | 0.59 | 0.62 | 0.95 |
| 16 | 0.65 | 0.37 | 0.40 | 0.51 |
| 17 | 0.62 | 0.45 | 0.43 | 0.75 |
| 18 | 0.53 | 0.30 | 0.37 | 0.46 |
| 19 | 0.49 | 0.29 | 0.42 | 0.81 |

On the Ryzen the best threaded transform is slower than the best serial one at
2^13 (1.48x) and 2^14 (1.21x) and breaks even at 2^15 (0.95x); on the Intel
notebook threads help from about 2^14.

**The change** (`src/otfft_algo.cpp` and the six `src/otfft_avx*.h`):

*   `OMP_THRESHOLD` per class: DIF4 2^13 to 2^18, DIT4 2^13 to 2^16, DIF8 2^11 to
    2^16, DIT8 2^13 to 2^16, DIF16 2^12 to 2^15, DIT16 2^11 to 2^14; SixStep stays
    at 2^14. The radix-4 and radix-8 classes break even at 2^18 to 2^20 on the
    Ryzen but at 2^14 to 2^16 on the Intel notebook; 2^16 is a compromise
    (removes the slowdowns of 1.5x to 3.2x below it, keeps most of the Intel gain,
    leaves 1.0x to 1.4x losses at 2^16 to 2^17 on the Ryzen).
*   With more than one thread the threaded table of D14 is used only from 2^15
    (`THREADED_MIN_LOG`); below it the serial table, and the classes of it do not
    use threads there.

**Effect on the Intel notebook (WSL2, GCC 15.2)**, `otfft_bench` of `master` and
of the branch, alternately, three runs each, medians, MFLOPS, complex
transforms; old / new:

| N | 2 threads | 4 threads | 8 threads |
| ---: | ---: | ---: | ---: |
| 1,024 | 36,180 / 34,481 | 35,850 / 37,145 | 37,149 / 35,302 |
| 4,096 | 22,734 / 24,308 | 23,170 / 25,035 | 22,895 / 24,145 |
| 16,384 | 20,456 / 28,155 | 149 / 26,822 | 72 / 25,699 |
| 65,536 | 16,288 / 16,407 | 28,784 / 26,718 | 18,066 / 315 |
| 262,144 | 16,441 / 15,073 | 29,396 / 29,701 | 20,049 / 20,429 |
| 1,048,576 | 13,289 / 12,981 | 25,809 / 25,518 | 6,183 / 6,190 |
| 4,194,304 | 9,838 / 10,359 | 19,228 / 19,073 | 19,884 / 18,474 |

*   N=16,384 is the case of the issue. With 4 and 8 threads the old code
    collapsed there (149 and 73 MFLOPS, one run of the old code even gave 52 or 76
    at 8 threads) while the new code gives 25,000 to 27,000, as with one thread;
    with 2 threads it is 1.38x faster. The old code behaved erratically there
    (a run at 8 threads gave 18,000 and the next 76).
*   N = 1,024, 4,096 and 262,144 to 4,194,304 are unchanged within the noise
    (0.92x to 1.08x).
*   N=65,536 with 8 threads is erratic in the unchanged code: with
    `OMP_WAIT_POLICY=active` the clean `master` build gave 852, 18,016 and 17,325
    MFLOPS in three runs, and the old and the new build both alternate between
    a few hundred and about 17,000. The plan chooses SixStep there in all
    builds, so the median of three runs in the table (316 for the new build, 18,066
    for the old) shows this erratic effect, not the thresholds (details in
    `docs/benchmark-data/2026-11-24-intel/threads/NOTES.txt`).

**Limits**

*   Not run on the Ryzen with the changed library: the thresholds for DIF4,
    DIT4, DIF8, DIT8 are a compromise between the two machines, based on
    one 12-thread run (with odd results at 2^19, 2^21 and 2^23, see below).
*   On this Intel notebook in WSL2, runs with 4 or 8 threads at small lengths
    were slow by factors of 40 to 2,800 and often erratic (a call takes
    milliseconds); on native Linux (Ryzen) the same cases cost 1.0x to 3.2x. This
    looks like the wake-up latency of the OpenMP threads in the VM and the
    hybrid CPU, so those numbers are not used for the thresholds, only to
    show that the old thresholds make it worse. The first length where a class
    threads can still show it after the change (for example DIT16 at 2^14).
*   Not done: scaling above the break-even (R1.6 asks for better than 1.8x at
    4M points), the thresholds of the eight-step (2^13 and 2^19) and of the mixed
    radix, real FFT, DCT and Bluestein code, and the choice of the algorithm for
    threads. SixStep does not scale for odd powers of 2 from 2^19 (ratio 0.93 to
    1.09 at 2^19, 2^21, 2^23 on the Ryzen), where the eight-step path is used;
    that is part of the threaded table retune of R1.8.
*   One CPU with 2 to 8 threads and one 12-thread run on another: the
    break-even depends on the thread count and the CPU (cross-CCD cost on the
    Ryzen). A run-time calibration would be the general solution.

Exp kernels (R1.9): effect on the Intel notebook
------------------------------------------------

Roadmap R1.9, decision D19. From N = 2^18 the classes DIF8, DIT8, DIF16 and DIT16 use
the "exp" kernels ported from upstream 11.6e (`src/otfft_avx{dif,dit}{8,16}exp.h`):
they compute the twiddle factors in registers and read no weight table. Below 2^18
nothing changed (the same code runs), so the sizes 2^15 to 2^17 in the tables show the
noise of the measurement. `init_Wq` and `init_Wr`, named in the roadmap item, are not
used anywhere in upstream 11.6e (only defined in `otfft_misc.h`), so they were not ported.

Method: `ffttune --backend avx2 --threads 1 --min-log 15 --max-log 22`, one library built
from `master` before the change and one with the exp kernels, run alternately, three runs
each, median of the three. The cells are the time of the old build divided by the time of
the new build for the same class (above 1: the new build is faster). Intel Core Ultra 7
268V, MSVC on Windows 11 and GCC 15.2 in WSL2 (`taskset -c 2`, one core). Raw logs in
`docs/benchmark-data/2026-11-24-intel/exp/`.

**Per class, 1 thread, old time / new time**

| log2 N | DIF8 MSVC | DIF8 GCC | DIT8 MSVC | DIT8 GCC | DIF16 MSVC | DIF16 GCC | DIT16 MSVC | DIT16 GCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 1.10 | 0.90 | 1.27 | 0.81 | 0.87 | 1.05 | 1.26 | 1.05 |
| 19 | 1.11 | 0.88 | 0.99 | 0.91 | 0.75 | 1.35 | 1.20 | 1.17 |
| 20 | 1.06 | 0.95 | 1.05 | 0.96 | 0.81 | 1.28 | 1.08 | 1.15 |
| 21 | 1.13 | 0.92 | 1.17 | 0.96 | 1.01 | 1.35 | 1.28 | 1.21 |
| 22 | 1.09 | 0.92 | 1.10 | 0.95 | 0.94 | 1.33 | 1.16 | 1.18 |
| noise (2^15 to 2^17, same code) | 0.94 to 1.05 | 0.79 to 1.10 | 0.96 to 0.97 | 0.97 to 1.12 | 0.93 to 0.99 | 0.98 to 1.20 | 0.92 to 1.04 | 0.95 to 1.02 |

**Best class per length (the one `ffttune` would pick), 1 thread**

| log2 N | MSVC old best | MSVC new best | MSVC ratio | GCC old best | GCC new best | GCC ratio |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 18 | DIF8 | DIF8 | 1.10 | DIT8 | DIT16 | 0.93 |
| 19 | DIF16 | DIF8 | 1.01 | DIT8 | DIT16 | 1.03 |
| 20 | DIF8 | DIF8 | 1.06 | DIT8 | DIT16 | 1.03 |
| 21 | DIF8 | DIF8 | 1.13 | DIF8 | DIT8 | 0.93 |
| 22 | DIF8 | DIF8 | 1.09 | DIT8 | DIT16 | 1.08 |

With 8 threads (MSVC, `ffttune --threads 8`) the best class is 1.36x faster at 2^18, 1.09x
at 2^19, 1.02x at 2^20, 0.99x at 2^21 and 0.97x at 2^22.

What this shows:

*   The exp kernels are correct (`otfft_exp_test`, all backends, all six functions) and
    **do not change the speed of the best class by more than the noise** on this CPU:
    0.93 to 1.13 with one thread, depending on the compiler.
*   The effect depends on the class and on the compiler. DIT16 is faster on both
    compilers (1.05 to 1.28). DIF16 is faster with GCC (1.05 to 1.35) and slower with MSVC
    at 2^18 to 2^20 (0.75 to 0.87). DIF8 and DIT8 are faster with MSVC (up to 1.27) and a
    little slower with GCC (0.81 to 0.96).
*   The large gain that upstream's table showed for N = 2^16 to 2^20 (+28% to +94%, see
    `docs/upstream-comparison.md`) came from the algorithm table (R1.8), not from the exp
    kernels. The upstream table picks DIF16 for most lengths above 2^15; with the exp kernels that choice is
    where GCC gains most here.

Limits:

*   One CPU. The target of the item is the Ryzen (GCC), where the fork is 1.6x to 3x behind
    FFTW at large N; it was not run there. A prompt for that run is in the pull request.
*   The algorithm table was **not** changed. Under GCC the table (DIF8 for 2^18 to 2^21,
    tuned with the old kernels) now picks a class that is 5% to 12% slower than DIF16 or
    DIT16 at 2^19 to 2^22; the table has to be tuned again for the new kernels (R1.8) on
    more than one machine and compiler. At 2^15 to 2^17 the noise reached 0.79 to 1.20, so
    differences below about 20% in single cells are not reliable.
*   Three runs per cell, no end-to-end measurement through `createComplexFFT` with the
    library table and no run with the real FFT, DCT or Bluestein code that use these classes.
*   The exp kernels start an OpenMP parallel region on every call, also with one thread.
    The cost of that was not separated from the effect of the kernels.

OpenMP structure of the eight-step and six-step code (R1.10): effect on the Intel notebook
------------------------------------------------------------------------------------------

Roadmap R1.10, issue #19, decision D20. The power-of-two class SixStep runs the square
six-step code for even log2 N and the eight-step code for odd log2 N. Three changes from
upstream 11.6e: (1) the eight-step code from N = 2^19 runs in **one** `omp parallel` region;
(2) the six-step square code has a static schedule from 2^14 and a guided schedule above a
second threshold; (3) the eight-step code uses `omp sections` for its eight sub-transforms
below 2^19.

**The bug behind (1).** From 2^19 the fork called the six-step `part2` eight times *outside*
any parallel region. `part2` consists of orphaned `omp for` loops, which run on one thread
when no region is open, so most of the eight-step transform ran serially while the other
threads waited. That is the "SixStep does not scale at odd powers of 2 from 2^19" seen on the
Ryzen (ratio 0.93 to 1.09 with 12 threads at 2^19, 2^21, 2^23). Upstream wraps the whole
step in one region.

Method: `ffttune --backend avx2 --threads T` (T = 4 and 8) and `otfft_bench --kinds
complex,real,dct --sizes pow2 --min-n 65536 --lib OTFFT --threads 8`, a library built from
`master` before the change against the changed one, alternately, 3 runs each, medians; MSVC,
native Windows (Core Ultra 7 268V, 8 cores). Raw data in
`docs/benchmark-data/2026-11-24-intel/omp/`.

**SixStep class, odd log2 N (eight-step code), time old / time new**

| log2 N | 4 threads | 8 threads |
| ---: | ---: | ---: |
| 19 | 9,331 us -> 4,112 us (2.27x) | 9,401 -> 6,294 (1.49x) |
| 21 | 41,013 -> 16,733 (2.45x) | 40,433 -> 21,664 (1.87x) |
| 23 | 237,009 -> 151,628 (1.56x) | 256,416 -> 126,689 (2.02x) |

The run-to-run ranges do not overlap (2^21 with 4 threads: 39,983 to 43,328 old, 16,652 to
18,771 new). Below 2^19 (2^14 to 2^17, sections instead of an `omp for` over the eight
blocks) nothing measurable changed (0.86x to 1.42x with wide ranges; 0.98x in the repeat).

**SixStep class, even log2 N (six-step code, guided schedule from 2^18)**

| log2 N | 4 threads | 8 threads |
| ---: | ---: | ---: |
| 18 | 0.84x, 0.96x | 1.27x, 1.11x |
| 20 | 0.91x | 1.13x |
| 22 | 1.14x | 1.44x |

(Two numbers: the first full run and the repeat after the threshold change below.) With 4
threads the guided schedule is neutral to slightly slower at 2^18 to 2^20, with 8 threads
faster; upstream's own threshold is 2^16, where the guided schedule was **0.70x (8 threads)
and 0.81x (4 threads) of the static speed**, so the threshold is 2^17 here, which means
"from 2^18" (the square code exists for even log2 N only). After that change SixStep at 2^16
is 0.98x (8 threads) and 1.10x (4 threads) of the old speed, the same code as before.

**Through the library (`otfft_bench`, 8 threads, MFLOPS, final build against `master`)**

| N | complex | real | DCT |
| ---: | ---: | ---: | ---: |
| 65,536 | 1.14x | 1.01x | 1.01x |
| 262,144 | 0.98x | 0.90x | 1.10x |
| 1,048,576 | 1.09x | 1.19x | 1.38x |
| 4,194,304 | 1.19x | 1.82x | 1.58x |

65,536 is the median of 15 alternating pairs; the other rows are 3 runs, with ranges up to
1.4x at 262,144 (real: 13,877 to 19,156 MFLOPS old, 12,439 to 18,470 new; 2.9x in the
first round), so 0.90x to 1.10x there is noise. The threaded table of D14 uses SixStep from 2^19 up, which is where the gains
are; at N = 4,194,304 the real FFT goes from 8,397 to 15,253 MFLOPS.

What this shows:

*   The eight-step code now scales: 1.5x to 2.5x faster SixStep from 2^19 with 4 and 8 threads.
    That is the item "scale better above the break-even" of R1.6 and issue #19 for the lengths
    where the library uses SixStep. Complex N = 4,194,304 on 8 threads: 1.19x, real 1.82x,
    DCT 1.58x.
*   The best class per length mostly did not change (DIF8 wins most large lengths on this CPU
    with 4 and 8 threads, before and after), so the library gain depends on the table: SixStep
    is still **not** the fastest class at 2^19 to 2^23 with 4 or 8 threads here (DIF8 or DIF16
    are, by 1.1x to 2.1x). The threaded table (the upstream table) is therefore still wrong for
    this CPU above 2^19 (R1.8); the gain above is the gain for a table that picks SixStep.

Limits:

*   One CPU with 4 and 8 threads, MSVC. Not run on the Ryzen with 12 or 16 threads, where the
    problem was seen, and not with GCC (the build passes with GCC 15.2, see the pull request).
*   The target in the roadmap item (real and DCT with 12 threads from N = 262,144 at 0.5 to 0.8
    of upstream) was not compared with upstream again; at 262,144 the library did not change
    measurably.
*   The thresholds 2^14 (static) and 2^17 (guided) are from this CPU. The guided threshold is
    a deviation from upstream (2^16).
*   `firstprivate` clauses on the parallel regions (fork specific, not in upstream) are kept.

Problems found
--------------

1.  **Prime lengths with `createComplexFFT`** ([#17](https://github.com/DEWETRON/otfft/issues/17),
    [#16](https://github.com/DEWETRON/otfft/issues/16)). Only radix 2, 3, 4, 5 and 8 have
    butterflies; every other factor uses a naive O(k^2) DFT (`fwdfftany` in
    `src/otfft_mixedradix.h`). That gives 12 MFLOPS at N=1,009 and 2 MFLOPS at
    N=10,007 (434 ms), and 48 s at N=100,003. From a prime factor of 46,341
    the index `(i*j)%k` overflows `int`, so the result is wrong (correct at
    N=30,011, wrong at 50,021) and the twiddle table is read at a negative
    offset. The unit tests did not cover primes. **Fixed** (roadmap R1.4,
    D13): lengths with a prime factor above 100 now use Bluestein's algorithm.
    One run on the Intel notebook above (MSVC, one thread): N=1,009 gives
    2,069 MFLOPS (was 39), N=10,007 gives 1,110 (was 5) and N=100,003 is correct
    and gives 901 (was wrong). The real FFT and DCT are not routed yet.
2.  **Factor 7** ([#18](https://github.com/DEWETRON/otfft/issues/18)). N=10,080 is 11x
    slower than FFTW and 3.4x slower than PocketFFT, much worse than
    neighbouring sizes. There was no radix-7 butterfly; 7 went to the generic
    code above. **Fixed** (roadmap R1.5, D15): there is a radix-7 butterfly
    now; N=10,080 is 2.0x faster on the Intel notebook and faster than
    PocketFFT. The radix 3 and 5 butterflies are still plain; see "Radix-7
    butterfly: effect on the Intel notebook".
3.  **Threading** ([#19](https://github.com/DEWETRON/otfft/issues/19)). OpenMP is used for N=16,384 where it makes OTFFT 2.5x
    slower, and scales to only 1.8x at N=4,194,304. **Partly fixed** (roadmap R1.6,
    D17): the class thresholds are at the measured break-even now and the
    serial table is used below 2^15 with several threads; the scaling above is
    not improved yet. See "Thread break-even and the OpenMP thresholds".
    The exp kernels of roadmap R1.9 (large N) are ported; on the Intel notebook they
    change the best class by 0.93x to 1.13x, see "Exp kernels (R1.9)". The eight-step
    code from N = 2^19 ran its six-step parts on one thread (a missing `omp parallel`);
    fixed in R1.10: SixStep 1.5x to 2.5x faster there, complex N = 4,194,304 with 8
    threads 1.19x, real 1.82x, DCT 1.58x (see "OpenMP structure of the eight-step and
    six-step code (R1.10)").
4.  **AVX2.** No gain over AVX in these tests, so the AVX2 backend adds
    build time for no measured benefit.
5.  **DCT scale.** OTFFT's DCT-II output is 0.5 times FFTW's and PocketFFT's
    (they compute 2 * sum, OTFFT computes sum). Not a bug, but callers moving
    between libraries need to know.

Caveats
-------

*   The CPU governor was `schedutil` and other work was running. Sizes at N >= 1,048,576
    are memory-bound and vary by up to 30% between runs (FFTW-estimate at
    N=4,194,304 gave 2,054 to 4,976 MFLOPS in separate one-thread runs). Treat
    differences under 30% there as noise.
*   FFTW's planning was limited to 10 s, so `FFTW_MEASURE` plans for N >=
    1,048,576 may be worse than an unlimited search. At N=4,194,304 it even
    measured slower than `FFTW_ESTIMATE`.
*   One machine, one compiler, one run per setting. No ARM64 or AVX-512 numbers
    yet (roadmap R2.1 and R2.2).
*   PocketFFT is called through its high-level API, which looks up its plan cache
    and allocates temporary memory on each call. A hand-managed plan could be
    faster for small N.
*   FFTW is GPL licensed. It is linked only into the benchmark tool, never into
    the OTFFT library, and is not vendored in this repository.

Reproduce
---------

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DOTFFT_BUILD_BENCHMARKS=ON
cmake --build build --parallel --target otfft_bench
build/Release/otfft_bench --threads 1  --csv bench-t1.csv
build/Release/otfft_bench --threads 12 --csv bench-t12.csv
python3 benchmarks/report.py bench-t1.csv bench-t12.csv > tables.md
```

FFTW is found with pkg-config (`libfftw3-dev` on Debian and Ubuntu); its
OpenMP threading library `libfftw3_omp` is used when it is found next to it.
Without FFTW the benchmark runs the other libraries. To compare with an unpacked
upstream OTFFT release, configure with `-DOTFFT_UPSTREAM_DIR=<path to otfft-11.6e>`
and run `otfft_bench_upstream` with the same options; `benchmarks/compare_upstream.py`
turns pairs of CSV files into the ratio tables above. `--min-n N` skips
smaller sizes. Options: `--kinds complex,real,dct,roundtrip`,
`--sizes pow2,pow2all,smooth,prime`, `--lib NAME`, `--samples S`, `--quick`.

The raw results and logs of this run are in
`docs/benchmark-data/2026-11-23/`; the upstream comparison is in
`docs/benchmark-data/2026-11-24/` (`up-*` main runs, `large-*` repeats).
