// Placement of x, y (work buffer) and W (twiddle table) in memory: time of the forward mixed radix
// transform as a function of the offsets modulo 4096 bytes.
// usage: place N [step]
#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"
#include "otfft_avx2.h"
#include "otfft_mixedradix.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace OTFFT_NAMESPACE;
using namespace OTFFT_NAMESPACE::OTFFT_MixedRadix;
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); }

static double timeit(int N, complex_t* x, complex_t* y, complex_t* W)
{
    double best = 1e30;
    for (int rep = 0; rep < 25; ++rep) {
        const double t0 = now();
        fwdfft(N, 1, 0, x, y, W);
        best = std::min(best, now() - t0);
    }
    return best;
}

int main(int argc, char** argv)
{
    // usage: place1 N step passes   (dy sweep only, W at a fixed offset)
    const int N = atoi(argv[1]);
    const int step = atoi(argv[2]);
    const int passes = atoi(argv[3]);
    const size_t bytes = (size_t)N * 16;
    const size_t span = ((bytes + 4096 + 8191) / 4096) * 4096;
    char* base = (char*)aligned_alloc(4096, span * 3 + 4096);
    memset(base, 1, span * 3 + 4096);
    const double th = 2 * M_PI / N;
    complex_t* x = (complex_t*)(base);
    complex_t* W = (complex_t*)(base + 2 * span + 1024);
    for (int p = 0; p <= N; ++p) W[p] = complex_t(cos(p * th), -sin(p * th));
    for (int i = 0; i < N; ++i) x[i] = complex_t((i * 7) % 11 - 5.0, (i * 5) % 13 - 6.0);
    const int n = 4096 / step;
    std::vector<double> t(n, 1e30);
    for (int pass = 0; pass < passes; ++pass)
        for (int a = 0; a < n; ++a) {
            complex_t* y = (complex_t*)(base + span + a * step);
            t[a] = std::min(t[a], timeit(N, x, y, W));
        }
    double lo = 1e30; for (double v : t) lo = std::min(lo, v);
    printf("N=%d step=%d passes=%d best %.1f us\n", N, step, passes, lo);
    for (int a = 0; a < n; ++a) printf("%d %.1f %.2f\n", a * step, t[a], t[a] / lo);
}
