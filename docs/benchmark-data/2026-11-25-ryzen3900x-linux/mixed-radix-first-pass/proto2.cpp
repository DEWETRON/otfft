// Prototype: s == 1 radix-8 and radix-4 passes, two values of p at once.
#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"
#include "otfft_avx2.h"
#include "otfft_mixedradix.h"
#include <chrono>
#include <cmath>
#include <cstdio>
using namespace OTFFT_NAMESPACE;
using namespace OTFFT_NAMESPACE::OTFFT_MixedRadix;
using clk = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); }
template <class F> static double timeit(F f)
{
    double best = 1e30;
    for (int rep = 0; rep < 9; ++rep) {
        int iters = 1; double t;
        for (;;) { double t0 = now(); for (int i = 0; i < iters; ++i) f(); t = now() - t0; if (t > 3000) break; iters *= 2; }
        best = std::min(best, t / iters);
    }
    return best;
}
// store the two lanes of v to y0 (low) and y1 (high)
static inline void store2(complex_vector y0, complex_vector y1, const ymm v)
{
    _mm_storeu_pd(&y0->Re, _mm256_castpd256_pd128(v));
    _mm_storeu_pd(&y1->Re, _mm256_extractf128_pd(v, 1));
}

static inline ymm ld2(const_complex_vector W, int k, int p) { return _mm256_loadu2_m128d(&W[k*(p+1)].Re, &W[k*p].Re); }
static void core8_s1(const int n, complex_vector x, complex_vector y, const_complex_vector W)
{
    const int m = n / 8, N1 = n / 8;
    int p = 0;
    for (; p + 1 < m; p += 2) {
        complex_vector x_p = x + p, y_8p = y + 8 * p;
        const ymm w1p = getpz2(W + p);
        const ymm w2p = mulpz2(w1p, w1p), w3p = mulpz2(w1p, w2p), w4p = mulpz2(w2p, w2p);
        const ymm w5p = mulpz2(w2p, w3p), w6p = mulpz2(w3p, w3p), w7p = mulpz2(w3p, w4p);
        const ymm x0 = getpz2(x_p), x1 = getpz2(x_p + N1), x2 = getpz2(x_p + N1 * 2), x3 = getpz2(x_p + N1 * 3);
        const ymm x4 = getpz2(x_p + N1 * 4), x5 = getpz2(x_p + N1 * 5), x6 = getpz2(x_p + N1 * 6), x7 = getpz2(x_p + N1 * 7);
        const ymm  a04 = addpz2(x0, x4), s04 = subpz2(x0, x4);
        const ymm  a26 = addpz2(x2, x6), js26 = jxpz2(subpz2(x2, x6));
        const ymm  a15 = addpz2(x1, x5), s15 = subpz2(x1, x5);
        const ymm  a37 = addpz2(x3, x7), js37 = jxpz2(subpz2(x3, x7));
        const ymm a04_p1_a26 = addpz2(a04, a26), s04_mj_s26 = subpz2(s04, js26);
        const ymm a04_m1_a26 = subpz2(a04, a26), s04_pj_s26 = addpz2(s04, js26);
        const ymm a15_p1_a37 = addpz2(a15, a37);
        const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
        const ymm j_a15_m1_a37 = jxpz2(subpz2(a15, a37));
        const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
        store2(y_8p + 0, y_8p + 8 + 0, addpz2(a04_p1_a26, a15_p1_a37));
        store2(y_8p + 1, y_8p + 8 + 1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
        store2(y_8p + 2, y_8p + 8 + 2, mulpz2(w2p, subpz2(a04_m1_a26, j_a15_m1_a37)));
        store2(y_8p + 3, y_8p + 8 + 3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
        store2(y_8p + 4, y_8p + 8 + 4, mulpz2(w4p, subpz2(a04_p1_a26, a15_p1_a37)));
        store2(y_8p + 5, y_8p + 8 + 5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
        store2(y_8p + 6, y_8p + 8 + 6, mulpz2(w6p, addpz2(a04_m1_a26, j_a15_m1_a37)));
        store2(y_8p + 7, y_8p + 8 + 7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
    }
    if (p < m) fwdcore8(n, 1, x, y, W);
}

static void core8_s1b(const int n, complex_vector x, complex_vector y, const_complex_vector W)
{
    const int m = n / 8, N1 = n / 8;
    int p = 0;
    for (; p + 1 < m; p += 2) {
        complex_vector x_p = x + p, y_8p = y + 8 * p;
        const ymm w1p = getpz2(W + p);
        const ymm w2p = ld2(W, 2, p), w3p = ld2(W, 3, p), w4p = ld2(W, 4, p);
        const ymm w5p = ld2(W, 5, p), w6p = ld2(W, 6, p), w7p = ld2(W, 7, p);
        const ymm x0 = getpz2(x_p), x1 = getpz2(x_p + N1), x2 = getpz2(x_p + N1 * 2), x3 = getpz2(x_p + N1 * 3);
        const ymm x4 = getpz2(x_p + N1 * 4), x5 = getpz2(x_p + N1 * 5), x6 = getpz2(x_p + N1 * 6), x7 = getpz2(x_p + N1 * 7);
        const ymm  a04 = addpz2(x0, x4), s04 = subpz2(x0, x4);
        const ymm  a26 = addpz2(x2, x6), js26 = jxpz2(subpz2(x2, x6));
        const ymm  a15 = addpz2(x1, x5), s15 = subpz2(x1, x5);
        const ymm  a37 = addpz2(x3, x7), js37 = jxpz2(subpz2(x3, x7));
        const ymm a04_p1_a26 = addpz2(a04, a26), s04_mj_s26 = subpz2(s04, js26);
        const ymm a04_m1_a26 = subpz2(a04, a26), s04_pj_s26 = addpz2(s04, js26);
        const ymm a15_p1_a37 = addpz2(a15, a37);
        const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
        const ymm j_a15_m1_a37 = jxpz2(subpz2(a15, a37));
        const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
        store2(y_8p + 0, y_8p + 8 + 0, addpz2(a04_p1_a26, a15_p1_a37));
        store2(y_8p + 1, y_8p + 8 + 1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
        store2(y_8p + 2, y_8p + 8 + 2, mulpz2(w2p, subpz2(a04_m1_a26, j_a15_m1_a37)));
        store2(y_8p + 3, y_8p + 8 + 3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
        store2(y_8p + 4, y_8p + 8 + 4, mulpz2(w4p, subpz2(a04_p1_a26, a15_p1_a37)));
        store2(y_8p + 5, y_8p + 8 + 5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
        store2(y_8p + 6, y_8p + 8 + 6, mulpz2(w6p, addpz2(a04_m1_a26, j_a15_m1_a37)));
        store2(y_8p + 7, y_8p + 8 + 7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
    }
    if (p < m) {   // odd m: last p with the scalar code
        fwdcore8(n, 1, x, y, W);   // reference path, used only to test odd m in this prototype
    }
}

static void core4_s1(const int n, complex_vector x, complex_vector y, const_complex_vector W)
{
    const int m = n / 4, N1 = n / 4;
    int p = 0;
    for (; p + 1 < m; p += 2) {
        complex_vector x_p = x + p, y_4p = y + 4 * p;
        const ymm w1p = getpz2(W + p), w2p = ld2(W, 2, p), w3p = ld2(W, 3, p);
        const ymm a = getpz2(x_p), b = getpz2(x_p + N1), c = getpz2(x_p + N1 * 2), d = getpz2(x_p + N1 * 3);
        const ymm apc = addpz2(a, c), amc = subpz2(a, c), bpd = addpz2(b, d), jbmd = jxpz2(subpz2(b, d));
        store2(y_4p + 0, y_4p + 4 + 0, addpz2(apc, bpd));
        store2(y_4p + 1, y_4p + 4 + 1, mulpz2(w1p, subpz2(amc, jbmd)));
        store2(y_4p + 2, y_4p + 4 + 2, mulpz2(w2p, subpz2(apc, bpd)));
        store2(y_4p + 3, y_4p + 4 + 3, mulpz2(w3p, addpz2(amc, jbmd)));
    }
    for (; p < m; p++) {
        complex_vector x_p = x + p, y_4p = y + 4 * p;
        const xmm w1p = getpz(W[p]);
        const xmm w2p = mulpz(w1p,w1p);
        const xmm w3p = mulpz(w1p,w2p);
        const xmm a = getpz(x_p[0]), b = getpz(x_p[N1]), c = getpz(x_p[N1*2]), d = getpz(x_p[N1*3]);
        const xmm apc = addpz(a, c), amc = subpz(a, c), bpd = addpz(b, d), jbmd = jxpz(subpz(b, d));
        setpz(y_4p[0], addpz(apc, bpd));
        setpz(y_4p[1], mulpz(w1p, subpz(amc, jbmd)));
        setpz(y_4p[2], mulpz(w2p, subpz(apc, bpd)));
        setpz(y_4p[3], mulpz(w3p, addpz(amc, jbmd)));
    }
}

int main()
{
    struct Case { int R, n; };
    const Case cases[] = {{8, 2520}, {8, 5040}, {8, 10080}, {8, 100000}, {8, 64}, {8, 24}, {4, 1260}, {4, 12}, {4, 100}, {4, 4 * 25 * 49 * 9}};
    for (const Case& c : cases) {
        const int N = c.n;
        simd_array<complex_t> xa(N), y1a(N), y2a(N), wa(N + 1);
        complex_t *x = &xa, *y1 = &y1a, *y2 = &y2a, *W = &wa;
        const double th = 2 * M_PI / N;
        for (int p = 0; p <= N; ++p) W[p] = complex_t(cos(p * th), -sin(p * th));
        for (int i = 0; i < N; ++i) { x[i] = complex_t((i * 7) % 11 - 5.0, (i * 5) % 13 - 6.0); y1[i] = y2[i] = complex_t(0, 0); }
        auto ref = [&] { if (c.R == 8) fwdcore8(N, 1, x, y1, W); else fwdcore4(N, 1, x, y1, W); };
        auto avx = [&] { if (c.R == 8) core8_s1(N, x, y2, W); else core4_s1(N, x, y2, W); };
        ref(); avx();
        double err = 0; for (int i = 0; i < N; ++i) err = std::max(err, std::max(std::fabs(y1[i].Re - y2[i].Re), std::fabs(y1[i].Im - y2[i].Im)));
        const bool oddm = (c.R == 8) && ((N / c.R) % 2);
        double t3 = 0, err3 = 0;
        if (c.R == 8 && !oddm) { core8_s1b(N, x, y1, W); fwdcore8(N,1,x,y2,W); for (int i = 0; i < N; ++i) err3 = std::max(err3, std::max(std::fabs(y1[i].Re - y2[i].Re), std::fabs(y1[i].Im - y2[i].Im))); t3 = timeit([&]{ core8_s1b(N, x, y1, W); }); }
        const double t1 = timeit(ref), t2 = oddm ? t1 : timeit(avx);
        if (t3 > 0) printf("   table twiddles: %.2f us (%.2fx), max diff to old %.1e\n", t3, t1 / t3, err3);
        printf("radix %d n=%-7d m=%-6d%s  max diff %.1e   now %8.2f us   ymm %8.2f us   %.2fx\n", c.R, N, N / c.R, oddm ? " (odd m)" : "         ", err, t1, t2, t1 / t2);
    }
}
