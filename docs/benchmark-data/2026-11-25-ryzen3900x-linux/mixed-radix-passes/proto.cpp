// Prototype: ymm (two q values) versions of fwdcore3/5/7, compared with the existing ones.
#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"
#include "otfft_avx2.h"
#include "otfft_mixedradix.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
static inline ymm cst(double c) { return _mm256_set1_pd(c); }
static inline ymm mjx2(ymm a) { return negpz2(jxpz2(a)); }

static inline void b3(const ymm x0, const ymm x1, const ymm x2, ymm* r)
{
    const ymm t1 = addpz2(x1, x2);
    const ymm a1 = addpz2(x0, mulpd2(cst(R3_C1), t1));
    const ymm b1 = mulpd2(cst(R3_S1), subpz2(x1, x2));
    r[0] = addpz2(x0, t1);
    r[1] = subpz2(a1, jxpz2(b1)); r[2] = addpz2(a1, jxpz2(b1));
}
static inline void b5(const ymm x0, const ymm x1, const ymm x2, const ymm x3, const ymm x4, ymm* r)
{
    const ymm c1 = cst(R5_C1), c2 = cst(R5_C2), s1 = cst(R5_S1), s2 = cst(R5_S2);
    const ymm t1 = addpz2(x1, x4), d1 = subpz2(x1, x4);
    const ymm t2 = addpz2(x2, x3), d2 = subpz2(x2, x3);
    r[0] = addpz2(x0, addpz2(t1, t2));
    const ymm a1 = addpz2(x0, addpz2(mulpd2(c1, t1), mulpd2(c2, t2)));
    const ymm a2 = addpz2(x0, addpz2(mulpd2(c2, t1), mulpd2(c1, t2)));
    const ymm b1 = addpz2(mulpd2(s1, d1), mulpd2(s2, d2));
    const ymm b2 = subpz2(mulpd2(s2, d1), mulpd2(s1, d2));
    r[1] = subpz2(a1, jxpz2(b1)); r[4] = addpz2(a1, jxpz2(b1));
    r[2] = subpz2(a2, jxpz2(b2)); r[3] = addpz2(a2, jxpz2(b2));
}
static inline void b7(const ymm x0, const ymm x1, const ymm x2, const ymm x3, const ymm x4, const ymm x5, const ymm x6, ymm* r)
{
    const ymm c1 = cst(R7_C1), c2 = cst(R7_C2), c3 = cst(R7_C3), s1 = cst(R7_S1), s2 = cst(R7_S2), s3 = cst(R7_S3);
    const ymm t1 = addpz2(x1, x6), d1 = subpz2(x1, x6);
    const ymm t2 = addpz2(x2, x5), d2 = subpz2(x2, x5);
    const ymm t3 = addpz2(x3, x4), d3 = subpz2(x3, x4);
    r[0] = addpz2(x0, addpz2(t1, addpz2(t2, t3)));
    const ymm a1 = addpz2(x0, addpz2(mulpd2(c1, t1), addpz2(mulpd2(c2, t2), mulpd2(c3, t3))));
    const ymm a2 = addpz2(x0, addpz2(mulpd2(c2, t1), addpz2(mulpd2(c3, t2), mulpd2(c1, t3))));
    const ymm a3 = addpz2(x0, addpz2(mulpd2(c3, t1), addpz2(mulpd2(c1, t2), mulpd2(c2, t3))));
    const ymm b1 = addpz2(mulpd2(s1, d1), addpz2(mulpd2(s2, d2), mulpd2(s3, d3)));
    const ymm b2 = subpz2(subpz2(mulpd2(s2, d1), mulpd2(s3, d2)), mulpd2(s1, d3));
    const ymm b3 = addpz2(subpz2(mulpd2(s3, d1), mulpd2(s1, d2)), mulpd2(s2, d3));
    r[1] = subpz2(a1, jxpz2(b1)); r[6] = addpz2(a1, jxpz2(b1));
    r[2] = subpz2(a2, jxpz2(b2)); r[5] = addpz2(a2, jxpz2(b2));
    r[3] = subpz2(a3, jxpz2(b3)); r[4] = addpz2(a3, jxpz2(b3));
}

template <int R>
static void core_avx(const int n, const int s, complex_vector x, complex_vector y, const_complex_vector W)
{
    const int N = n * s, N1 = N / R, m = n / R;
    for (int p = 0; p < m; p++) {
        const int sp = s * p;
        ymm w[R];
        for (int k = 1; k < R; k++) w[k] = duppz2(getpz(W[k * sp]));
        for (int q = 0; q < s; q += 2) {
            const int q_sp = q + sp;
            ymm r[R];
            if (R == 3) b3(getpz2(x + q_sp), getpz2(x + q_sp + N1), getpz2(x + q_sp + N1 * 2), r);
            if (R == 5) b5(getpz2(x + q_sp), getpz2(x + q_sp + N1), getpz2(x + q_sp + N1 * 2), getpz2(x + q_sp + N1 * 3), getpz2(x + q_sp + N1 * 4), r);
            if (R == 7) b7(getpz2(x + q_sp), getpz2(x + q_sp + N1), getpz2(x + q_sp + N1 * 2), getpz2(x + q_sp + N1 * 3),
                           getpz2(x + q_sp + N1 * 4), getpz2(x + q_sp + N1 * 5), getpz2(x + q_sp + N1 * 6), r);
            const int o = q + sp * R;
            setpz2(y + o, r[0]);
            for (int k = 1; k < R; k++) setpz2(y + o + s * k, mulpz2(r[k], w[k]));
        }
    }
}

int main()
{
    struct Case { int R, n, s; };
    const Case cases[] = {{3, 63, 160}, {3, 21, 480}, {5, 315, 32}, {5, 625, 160}, {5, 25, 4000}, {7, 49, 4}, {7, 63, 160}, {3, 9, 2}};
    for (const Case& c : cases) {
        const int N = c.n * c.s;
        simd_array<complex_t> xa(N), y1a(N), y2a(N), wa(N + 1);
        complex_t *x = &xa, *y1 = &y1a, *y2 = &y2a, *W = &wa;
        const double th = 2 * M_PI / N;
        for (int p = 0; p <= N; ++p) W[p] = complex_t(cos(p * th), -sin(p * th));
        for (int i = 0; i < N; ++i) { x[i] = complex_t((i * 7) % 11 - 5.0, (i * 5) % 13 - 6.0); y1[i] = y2[i] = complex_t(0, 0); }
        auto ref = [&] { if (c.R == 3) fwdcore3(c.n, c.s, x, y1, W); else if (c.R == 5) fwdcore5(c.n, c.s, x, y1, W); else fwdcore7(c.n, c.s, x, y1, W); };
        auto avx = [&] { if (c.R == 3) core_avx<3>(c.n, c.s, x, y2, W); else if (c.R == 5) core_avx<5>(c.n, c.s, x, y2, W); else core_avx<7>(c.n, c.s, x, y2, W); };
        ref(); avx();
        double err = 0; for (int i = 0; i < N; ++i) err = std::max(err, std::max(std::fabs(y1[i].Re - y2[i].Re), std::fabs(y1[i].Im - y2[i].Im)));
        const double t1 = timeit(ref), t2 = timeit(avx);
        printf("radix %d n=%-5d s=%-5d N=%-7d  max diff %.1e   now %8.2f us   ymm %8.2f us   %.2fx\n", c.R, c.n, c.s, N, err, t1, t2, t1 / t2);
    }
}
