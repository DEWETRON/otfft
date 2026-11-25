// Time each pass of the forward mixed-radix FFT separately (single thread).
#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"
#include "otfft_avx2.h"
#include "otfft_mixedradix.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace OTFFT_NAMESPACE;
using namespace OTFFT_NAMESPACE::OTFFT_MixedRadix;
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); }

template <class F> static double timeit(F f)
{
    double best = 1e30;
    for (int rep = 0; rep < 9; ++rep) {
        int iters = 1;
        double t;
        for (;;) {
            double t0 = now();
            for (int i = 0; i < iters; ++i) f();
            t = now() - t0;
            if (t > 3000) break;
            iters *= 2;
        }
        best = std::min(best, t / iters);
    }
    return best;
}

int main(int argc, char** argv)
{
    for (int a = 1; a < argc; ++a) {
        const int N = atoi(argv[a]);
        simd_array<complex_t> xa(N), ya(N), wa(N + 1);
        complex_t* x = &xa; complex_t* y = &ya; complex_t* W = &wa;
        const double th = 2 * M_PI / N;
        for (int p = 0; p <= N; ++p) W[p] = complex_t(cos(p * th), -sin(p * th));
        for (int i = 0; i < N; ++i) { x[i] = complex_t(i % 7, i % 5); y[i] = complex_t(0, 0); }
        printf("N=%d\n", N);
        double total = 0;
        int n = N, s = 1;
        bool eo = false;
        while (n > 1) {
            int r = n % 8 == 0 ? 8 : n % 4 == 0 ? 4 : n % 2 == 0 ? 2 : n % 5 == 0 ? 5 : n % 3 == 0 ? 3 : 7;
            double t;
            if (n == r) {
                t = timeit([&] { switch (r) { case 8: fwdend8(s, eo, x, y); break; case 4: fwdend4(s, eo, x, y); break;
                    case 2: fwdend2(s, eo, x, y); break; case 5: fwdend5(s, eo, x, y); break; case 3: fwdend3(s, eo, x, y); break;
                    default: fwdend7(s, eo, x, y); } });
            } else {
                t = timeit([&] { switch (r) { case 8: fwdcore8(n, s, x, y, W); break; case 4: fwdcore4(n, s, x, y, W); break;
                    case 2: fwdcore2(n, s, x, y, W); break; case 5: fwdcore5(n, s, x, y, W); break; case 3: fwdcore3(n, s, x, y, W); break;
                    default: fwdcore7(n, s, x, y, W); } });
            }
            printf("  radix %d  n=%-6d s=%-6d  %8.2f us  (%.2f ns/point)\n", r, n, s, t, 1000.0 * t / N);
            total += t;
            n /= r; s *= r; eo = !eo; std::swap(x, y);
        }
        printf("  sum of passes %.2f us\n", total);
    }
}
