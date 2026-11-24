// Prototype (not part of the repo): "few exchanges" complex FFT for N = n*n, forward, scale_1.
//   pass A: column FFTs (j1 direction) on strips of W adjacent columns, gathered into a thread-local
//           buffer, twiddle W_N^(j2*k1), written to y as row-major B[k1][j2]
//   pass C: row FFTs of y (in place, thread-local scratch), written transposed into x (full lines)
// Two passes over the array instead of five; ownership of data changes once (A -> C) and the
// output is written by the thread that computed it.
#include "/home/glaure/Development/otfft/src/otfft_algo_sixstep.cpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace OTFFT_AVX2 { namespace OTFFT_SixStep {

#ifndef FX_WS
#define FX_WS 4
#endif
constexpr int WS = FX_WS;   // strip width in complex numbers (4 = one 64 byte line)

static double now_us() { using namespace std::chrono; return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count(); }

template <int LOG> struct fewx
{
    static constexpr int N = 1 << LOG;
    static constexpr int n = 1 << (LOG / 2);

    // twiddle table in the order pass A reads it: [strip][k1][c], c < WS
    static void init_twiddle(simd_array<complex_t>& a)
    {
        a.setup(N);
        complex_t* t = &a;
        const double th = -2 * M_PI / N;
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < n / WS; s++)
            for (int k = 0; k < n; k++)
                for (int c = 0; c < WS; c++) {
                    const long long e = (long long)k * (s * WS + c) % N;
                    t[((size_t)s * n + k) * WS + c] = expj(th * (double)e);
                }
    }

    static void forward(complex_t* x, complex_t* y, const complex_t* Ws, const complex_t* Wt,
                        complex_t* const* bufs, complex_t* const* scr, double* tm = nullptr)
    {
        #pragma omp parallel
        {
            if (tm) {
                #pragma omp barrier
                #pragma omp master
                tm[0] = now_us();
            }
            const int tid = omp_get_thread_num();
            complex_t* buf = bufs[tid];   // WS * n
            complex_t* sc = scr[tid];     // n
            #pragma omp for schedule(static)
            for (int s = 0; s < n / WS; s++) {
                const int c0 = s * WS;
                for (int r = 0; r < n; r++) {
                    const complex_t* line = x + (size_t)r * n + c0;
                    for (int c = 0; c < WS; c++) setpz(buf[c * n + r], getpz(line[c]));
                }
                for (int c = 0; c < WS; c++)
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(buf + c * n, sc, Ws);
                const complex_t* tw = Wt + (size_t)s * n * WS;
                for (int k = 0; k < n; k++) {
                    for (int c = 0; c < WS; c += 2) {
                        const ymm v = cat(getpz(buf[c * n + k]), getpz(buf[(c + 1) * n + k]));
                        setpz2(y + (size_t)k * n + c0 + c, mulpz2(getpz2(tw + k * WS + c), v));
                    }
                }
            }
            if (tm) {
                #pragma omp barrier
                #pragma omp master
                tm[1] = now_us();
            }
            #pragma omp for schedule(static)
            for (int b = 0; b < n / 4; b++) {
                const int k0 = b * 4;
                for (int j = 0; j < 4; j++)
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(y + (size_t)(k0 + j) * n, sc, Ws);
                for (int k2 = 0; k2 < n; k2++) {
                    const complex_t* r0 = y + (size_t)(k0 + 0) * n + k2;
                    const ymm v01 = cat(getpz(r0[0]), getpz(r0[n]));
                    const ymm v23 = cat(getpz(r0[2 * n]), getpz(r0[3 * n]));
                    setpz2(x + (size_t)k2 * n + k0 + 0, v01);
                    setpz2(x + (size_t)k2 * n + k0 + 2, v23);
                }
            }
            if (tm) {
                #pragma omp barrier
                #pragma omp master
                tm[2] = now_us();
            }
        }
    }
};

}}

using namespace OTFFT_AVX2;
using namespace OTFFT_AVX2::OTFFT_SixStep;
typedef std::chrono::steady_clock Clock;

template <int LOG> int run(int reps, const char* mode)
{
    typedef fewx<LOG> F;
    const int N = F::N, n = F::n;
    const bool useNew = !std::strcmp(mode, "new");
    OTFFT_AVX2::OTFFT_SixStep::FFT0 ref; ref.setup2(LOG);
    simd_array<complex_t> aw, ax(N), ay(N), az(N);
    if (useNew) F::init_twiddle(aw);
    complex_t *x = &ax, *y = &ay;
    const int T = omp_get_max_threads();
    std::vector<simd_array<complex_t>> bv(T), sv(T);
    std::vector<complex_t*> bufs(T), scr(T);
    for (int t = 0; t < T; t++) { bv[t].setup(WS * n); sv[t].setup(n); bufs[t] = &bv[t]; scr[t] = &sv[t]; }
    auto fill = [&](complex_t* p) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) { p[i].Re = std::cos(0.001 * i) + 0.1 * (i % 7); p[i].Im = std::sin(0.002 * i) - 0.1 * (i % 5); }
    };
    if (!std::strcmp(mode, "check")) {
        F::init_twiddle(aw);
        fill(x); fill(&az[0]);
        F::forward(x, y, ref.Ws, &aw, bufs.data(), scr.data());
        ref.fwd0(&az[0], y);
        double err = 0, mag = 0;
        for (int i = 0; i < N; i++) { err = std::max(err, std::hypot(x[i].Re - az[i].Re, x[i].Im - az[i].Im)); mag = std::max(mag, std::hypot(az[i].Re, az[i].Im)); }
        std::printf("log2N=%d check: max abs error %.3g (max |X| %.3g)\n", LOG, err, mag);
        return err > 1e-9 * mag;
    }
    std::vector<double> t;
    for (int r = 0; r < reps + 3; r++) {
        fill(x);
        auto t0 = Clock::now();
        if (useNew) F::forward(x, y, ref.Ws, &aw, bufs.data(), scr.data()); else ref.fwd0(x, y);
        auto t1 = Clock::now();
        if (r >= 3) t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(t.begin(), t.end());
    std::printf("log2N=%d threads=%d mode=%s %.0f us\n", LOG, T, mode, t[t.size() / 2]);
    return 0;
}

int main(int argc, char** argv)
{
    int lg = argc > 1 ? std::atoi(argv[1]) : 18;
    const char* mode = argc > 2 ? argv[2] : "new";
    int reps = argc > 3 ? std::atoi(argv[3]) : 41;
    switch (lg) { case 16: return run<16>(reps, mode); case 18: return run<18>(reps, mode); case 20: return run<20>(reps, mode); case 22: return run<22>(reps, mode); }
    return 1;
}
