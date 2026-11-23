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
    to 2^24. A round trip like upstream's does not reproduce that here: see "Comparison
    with the upstream benchmark".
*   **Compilers.** GCC 15.2 and Clang 21.1.8 give the same picture; the real FFT and DCT
    are 4 to 12 percent slower with Clang and Bluestein 15 to 43 percent. See "Compiler comparison".
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
*   Possible reasons, **none of which is verified**: a different CPU (Zen 2 against
    a 2016 Intel core), compiler (GCC 15.2 against Clang 8) and FFTW version
    (3.3.10 against 3.3.8); this fork's merged OTFFT is not upstream 11.6 (see
    below); different build flags; and timing noise on a shared machine. A
    run on the author's hardware class (an Intel core with AVX2) would tell
    which of these matters.
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

Problems found
--------------

1.  **Prime lengths with `createComplexFFT`** ([#17](https://github.com/DEWETRON/otfft/issues/17),
    [#16](https://github.com/DEWETRON/otfft/issues/16)). Only radix 2, 3, 4, 5 and 8 have
    butterflies; every other factor uses a naive O(k^2) DFT (`fwdfftany` in
    `src/otfft_mixedradix.h`). That gives 12 MFLOPS at N=1,009 and 2 MFLOPS at
    N=10,007 (434 ms), and 48 s at N=100,003. From a prime factor of 46,341
    the index `(i*j)%k` overflows `int`, so the result is wrong (correct at
    N=30,011, wrong at 50,021) and the twiddle table is read at a negative
    offset. The unit tests do not cover primes. Use `createBluesteinFFT` for
    such lengths.
2.  **Factor 7** ([#18](https://github.com/DEWETRON/otfft/issues/18)). N=10,080 is 11x
    slower than FFTW and 3.4x slower than PocketFFT, much worse than
    neighbouring sizes. There is no radix-7 butterfly; 7 goes to the generic
    code above.
3.  **Threading** ([#19](https://github.com/DEWETRON/otfft/issues/19)). OpenMP is used for N=16,384 where it makes OTFFT 2.5x
    slower, and scales to only 1.8x at N=4,194,304.
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
Without FFTW the benchmark runs the other libraries. Options: `--kinds complex,real,dct,roundtrip`,
`--sizes pow2,pow2all,smooth,prime`, `--lib NAME`, `--samples S`, `--quick`.

The raw results and logs of this run are in
`docs/benchmark-data/2026-11-23/`.
