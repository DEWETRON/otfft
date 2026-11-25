Subject: OTFFT 11.6e: wrong result and O(N^2) time for lengths with a large prime factor

Hello,

while benchmarking OTFFT 11.6e (otfft-11.6e.tar.gz, 2023-12-21) we found two problems in
`otfft/otfft_mixedradix.h`, the generic radix code behind `OTFFT::FFT0` for lengths that are
not a product of 2, 3, 4, 5 and 8.

1. Integer overflow. The generic DFT indexes the twiddle table with `W[s*((i*j)%k)]`
   (and the three variants with `ms*` and `N-`). `i`, `j` and `k` are `int`. When the prime
   factor `k` is above 46,340, `i*j` can exceed 2^31, the modulo becomes negative and `W` is
   read at a negative offset. The result is wrong and depends on memory contents.
   The pattern is at lines 513, 533, 1070, 1090, 1650, 1675, 2234 and 2259.
   Using `(long long)i*j` or reducing before the multiplication fixes the index.

2. O(N^2) time. The same code is a naive DFT of size k with a modulo in the innermost loop,
   so a prime length takes quadratic time.

We measured this on the unmodified 11.6e sources (g++ -O1 -mavx -fopenmp, AMD Ryzen 9 3900X,
GCC 15.2, one thread, `OTFFT::FFT0(n).fwd0`, three bins compared with the direct sum):

| N (prime) | time | result |
| ---: | ---: | --- |
| 1,009 | 4 ms | correct (relative error 3.2e-14) |
| 10,007 | 428 ms | correct (3.3e-14) |
| 30,011 | 4,092 ms | correct (1.2e-14) |
| 50,021 | - | segmentation fault |
| 65,537 | - | segmentation fault |
| 100,003 | - | segmentation fault |

The time grows with N^2 (about 100x from 1,009 to 10,007). From N = 50,021 the program
crashes; in our fork, where the out-of-bounds read did not always crash, the result was wrong
(relative error 1.3 to 2.6 against a Bluestein transform).

We fixed it in our fork by routing lengths with a prime factor above 100 to Bluestein.

Best regards
