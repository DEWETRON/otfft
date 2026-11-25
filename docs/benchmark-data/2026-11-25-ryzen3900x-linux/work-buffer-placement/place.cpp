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
    const int N = atoi(argv[1]);
    const int step = argc > 2 ? atoi(argv[2]) : 64;
    const size_t bytes = (size_t)N * 16;
    const size_t span = ((bytes + 4096 + 8191) / 4096) * 4096;
    // three regions, each 4096-aligned, with room for the offset
    char* base = (char*)aligned_alloc(4096, span * 3 + 4096);
    memset(base, 1, span * 3 + 4096);
    const double th = 2 * M_PI / N;
    std::vector<double> t((4096 / step) * (4096 / step));
    // sweep: dy = (y - x) mod 4096, dW = (W - x) mod 4096
    int idx = 0;
    double tmin = 1e30, tmax = 0;
    for (int dy = 0; dy < 4096; dy += step)
        for (int dW = 0; dW < 4096; dW += step) {
            complex_t* x = (complex_t*)(base);
            complex_t* y = (complex_t*)(base + span + dy);
            complex_t* W = (complex_t*)(base + 2 * span + dW);
            for (int p = 0; p <= N; ++p) W[p] = complex_t(cos(p * th), -sin(p * th));
            for (int i = 0; i < N; ++i) { x[i] = complex_t((i * 7) % 11 - 5.0, (i * 5) % 13 - 6.0); }
            const double v = timeit(N, x, y, W);
            t[idx++] = v; tmin = std::min(tmin, v); tmax = std::max(tmax, v);
        }
    printf("N=%d  step=%d  best %.1f us  worst %.1f us  ratio %.2f\n", N, step, tmin, tmax, tmax / tmin);
    // per dy: mean and max over dW; per dW: mean over dy
    const int n = 4096 / step;
    printf("by dy (y - x mod 4096): min / max over dW\n");
    for (int a = 0; a < n; ++a) {
        double lo = 1e30, hi = 0;
        for (int b = 0; b < n; ++b) { lo = std::min(lo, t[a * n + b]); hi = std::max(hi, t[a * n + b]); }
        printf("  dy=%4d  %.1f .. %.1f\n", a * step, lo, hi);
    }
    printf("by dW (W - x mod 4096): min / max over dy\n");
    for (int b = 0; b < n; ++b) {
        double lo = 1e30, hi = 0;
        for (int a = 0; a < n; ++a) { lo = std::min(lo, t[a * n + b]); hi = std::max(hi, t[a * n + b]); }
        printf("  dW=%4d  %.1f .. %.1f\n", b * step, lo, hi);
    }
}
