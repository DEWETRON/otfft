/******************************************************************************
*  OTFFT Mixed Radix Version 11.4xv
*
*  Copyright (c) 2016 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_mixedradix_h
#define otfft_mixedradix_h

#include "otfft_misc.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_MixedRadix { //////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;

#ifdef DO_SINGLE_THREAD
constexpr int OMP_THRESHOLD = 1<<30;
#else
constexpr int OMP_THRESHOLD = 1<<15;
#endif

    struct cpx {
        xmm z;
        cpx(const xmm& z) noexcept : z(z) {}
        operator xmm() const noexcept { return z; }
    };

    static inline cpx operator+(const cpx a, const cpx b) noexcept
    {
        return addpz(a, b);
    }
    static inline cpx operator*(const cpx a, const cpx b) noexcept
    {
        return mulpz(a, b);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Radix-3 and radix-5 butterfly kernels
    //
    // The same symmetric form as the radix-7 kernel below: with the sums
    // t_j = x_j + x_(R-j) and the differences d_j = x_j - x_(R-j),
    //   X_k     = x_0 + sum_j cos(2 pi j k / R) t_j -+ i sum_j sin(2 pi j k / R) d_j
    //   X_(R-k) = x_0 + sum_j cos(2 pi j k / R) t_j +- i sum_j sin(2 pi j k / R) d_j
    // (upper sign: forward, lower: inverse). The multiplications are by real
    // constants, 2 for R = 3 and 8 for R = 5, instead of 4 and 16 complex ones.
    ///////////////////////////////////////////////////////////////////////////////

    constexpr double R3_C1 = -0.5;                    // cos(2 pi / 3)
    constexpr double R3_S1 =  0.86602540378443864676; // sin(2 pi / 3)
    constexpr double R5_C1 =  0.30901699437494742410; // cos(2 pi / 5)
    constexpr double R5_C2 = -0.80901699437494742410; // cos(4 pi / 5)
    constexpr double R5_S1 =  0.95105651629515357212; // sin(2 pi / 5)
    constexpr double R5_S2 =  0.58778525229247312917; // sin(4 pi / 5)
    constexpr double R7_C1 =  0.62348980185873353053; // cos(2 pi / 7)
    constexpr double R7_C2 = -0.22252093395631440429; // cos(4 pi / 7)
    constexpr double R7_C3 = -0.90096886790241912624; // cos(6 pi / 7)
    constexpr double R7_S1 =  0.78183148246802980871; // sin(2 pi / 7)
    constexpr double R7_S2 =  0.97492791218182360702; // sin(4 pi / 7)
    constexpr double R7_S3 =  0.43388373911755812048; // sin(6 pi / 7)

    // r[k] = k-th output; DIR = -1 forward, +1 inverse
    template <int DIR> static inline void bfly3(const xmm x0, const xmm x1, const xmm x2, xmm* r) noexcept force_inline3;
    template <int DIR> static inline void bfly3(const xmm x0, const xmm x1, const xmm x2, xmm* r) noexcept
    {
        const xmm c1 = cmplx(R3_C1, R3_C1);
        const xmm s1 = cmplx(R3_S1, R3_S1);
        const xmm t1 = addpz(x1, x2);
        const xmm a1 = addpz(x0, mulpd(c1, t1));
        const xmm b1 = mulpd(s1, subpz(x1, x2));
        r[0] = addpz(x0, t1);
        if (DIR < 0) { r[1] = addpz(a1, mjxpz(b1)); r[2] = addpz(a1, jxpz(b1)); }
        else         { r[1] = addpz(a1, jxpz(b1));  r[2] = addpz(a1, mjxpz(b1)); }
    }

    template <int DIR> static inline void bfly5(const xmm x0, const xmm x1, const xmm x2, const xmm x3, const xmm x4,
                                                xmm* r) noexcept force_inline3;
    template <int DIR> static inline void bfly5(const xmm x0, const xmm x1, const xmm x2, const xmm x3, const xmm x4,
                                                xmm* r) noexcept
    {
        const xmm c1 = cmplx(R5_C1, R5_C1);
        const xmm c2 = cmplx(R5_C2, R5_C2);
        const xmm s1 = cmplx(R5_S1, R5_S1);
        const xmm s2 = cmplx(R5_S2, R5_S2);
        const xmm t1 = addpz(x1, x4), d1 = subpz(x1, x4);
        const xmm t2 = addpz(x2, x3), d2 = subpz(x2, x3);
        r[0] = addpz(x0, addpz(t1, t2));
        const xmm a1 = addpz(x0, addpz(mulpd(c1, t1), mulpd(c2, t2)));
        const xmm a2 = addpz(x0, addpz(mulpd(c2, t1), mulpd(c1, t2)));
        const xmm b1 = addpz(mulpd(s1, d1), mulpd(s2, d2));
        const xmm b2 = subpz(mulpd(s2, d1), mulpd(s1, d2));
        if (DIR < 0) {
            r[1] = addpz(a1, mjxpz(b1)); r[4] = addpz(a1, jxpz(b1));
            r[2] = addpz(a2, mjxpz(b2)); r[3] = addpz(a2, jxpz(b2));
        }
        else {
            r[1] = addpz(a1, jxpz(b1)); r[4] = addpz(a1, mjxpz(b1));
            r[2] = addpz(a2, jxpz(b2)); r[3] = addpz(a2, mjxpz(b2));
        }
    }

#if defined(__AVX__) && defined(USE_INTRINSIC)
    ///////////////////////////////////////////////////////////////////////////////
    // AVX versions of the radix-3, 5 and 7 passes (roadmap R1.5, decision D28)
    //
    // Two values of q at once in one 256-bit register, for an even s. The butterflies
    // are those above, operation by operation, so the result is the same. With an odd s
    // (the first passes, when no factor 2 came before) the scalar passes are used.
    ///////////////////////////////////////////////////////////////////////////////

#define OTFFT_MIXED_YMM 1

    static inline ymm yconst(const double c) noexcept { return _mm256_set1_pd(c); }

    template <int R> struct BflyY;

    template <> struct BflyY<3> {
        template <int DIR> static inline void apply(const ymm* x, ymm* r) noexcept
        {
            const ymm t1 = addpz2(x[1], x[2]);
            const ymm a1 = addpz2(x[0], mulpd2(yconst(R3_C1), t1));
            const ymm b1 = mulpd2(yconst(R3_S1), subpz2(x[1], x[2]));
            r[0] = addpz2(x[0], t1);
            if (DIR < 0) { r[1] = subpz2(a1, jxpz2(b1)); r[2] = addpz2(a1, jxpz2(b1)); }
            else         { r[1] = addpz2(a1, jxpz2(b1)); r[2] = subpz2(a1, jxpz2(b1)); }
        }
    };

    template <> struct BflyY<5> {
        template <int DIR> static inline void apply(const ymm* x, ymm* r) noexcept
        {
            const ymm c1 = yconst(R5_C1), c2 = yconst(R5_C2), s1 = yconst(R5_S1), s2 = yconst(R5_S2);
            const ymm t1 = addpz2(x[1], x[4]), d1 = subpz2(x[1], x[4]);
            const ymm t2 = addpz2(x[2], x[3]), d2 = subpz2(x[2], x[3]);
            r[0] = addpz2(x[0], addpz2(t1, t2));
            const ymm a1 = addpz2(x[0], addpz2(mulpd2(c1, t1), mulpd2(c2, t2)));
            const ymm a2 = addpz2(x[0], addpz2(mulpd2(c2, t1), mulpd2(c1, t2)));
            const ymm b1 = addpz2(mulpd2(s1, d1), mulpd2(s2, d2));
            const ymm b2 = subpz2(mulpd2(s2, d1), mulpd2(s1, d2));
            if (DIR < 0) {
                r[1] = subpz2(a1, jxpz2(b1)); r[4] = addpz2(a1, jxpz2(b1));
                r[2] = subpz2(a2, jxpz2(b2)); r[3] = addpz2(a2, jxpz2(b2));
            }
            else {
                r[1] = addpz2(a1, jxpz2(b1)); r[4] = subpz2(a1, jxpz2(b1));
                r[2] = addpz2(a2, jxpz2(b2)); r[3] = subpz2(a2, jxpz2(b2));
            }
        }
    };

    template <> struct BflyY<7> {
        template <int DIR> static inline void apply(const ymm* x, ymm* r) noexcept
        {
            const ymm c1 = yconst(R7_C1), c2 = yconst(R7_C2), c3 = yconst(R7_C3);
            const ymm s1 = yconst(R7_S1), s2 = yconst(R7_S2), s3 = yconst(R7_S3);
            const ymm t1 = addpz2(x[1], x[6]), d1 = subpz2(x[1], x[6]);
            const ymm t2 = addpz2(x[2], x[5]), d2 = subpz2(x[2], x[5]);
            const ymm t3 = addpz2(x[3], x[4]), d3 = subpz2(x[3], x[4]);
            r[0] = addpz2(x[0], addpz2(t1, addpz2(t2, t3)));
            const ymm a1 = addpz2(x[0], addpz2(mulpd2(c1, t1), addpz2(mulpd2(c2, t2), mulpd2(c3, t3))));
            const ymm a2 = addpz2(x[0], addpz2(mulpd2(c2, t1), addpz2(mulpd2(c3, t2), mulpd2(c1, t3))));
            const ymm a3 = addpz2(x[0], addpz2(mulpd2(c3, t1), addpz2(mulpd2(c1, t2), mulpd2(c2, t3))));
            const ymm b1 = addpz2(mulpd2(s1, d1), addpz2(mulpd2(s2, d2), mulpd2(s3, d3)));
            const ymm b2 = subpz2(subpz2(mulpd2(s2, d1), mulpd2(s3, d2)), mulpd2(s1, d3));
            const ymm b3 = addpz2(subpz2(mulpd2(s3, d1), mulpd2(s1, d2)), mulpd2(s2, d3));
            if (DIR < 0) {
                r[1] = subpz2(a1, jxpz2(b1)); r[6] = addpz2(a1, jxpz2(b1));
                r[2] = subpz2(a2, jxpz2(b2)); r[5] = addpz2(a2, jxpz2(b2));
                r[3] = subpz2(a3, jxpz2(b3)); r[4] = addpz2(a3, jxpz2(b3));
            }
            else {
                r[1] = addpz2(a1, jxpz2(b1)); r[6] = subpz2(a1, jxpz2(b1));
                r[2] = addpz2(a2, jxpz2(b2)); r[5] = subpz2(a2, jxpz2(b2));
                r[3] = addpz2(a3, jxpz2(b3)); r[4] = subpz2(a3, jxpz2(b3));
            }
        }
    };

    // One butterfly of two values of q: x and y index as in the scalar passes
    // (x[q_sp + N1*k], y[q_sRp + s*k]); w[k] is the twiddle factor of output k.
    template <int R, int DIR>
    static inline void coreY(const int N1, const int s, const int q_sp, const int q_sRp, const ymm* w,
                             complex_vector x, complex_vector y) noexcept
    {
        ymm a[R], r[R];
        for (int k = 0; k < R; k++) a[k] = getpz2(x + (q_sp + N1*k));
        BflyY<R>::template apply<DIR>(a, r);
        setpz2(y + q_sRp, r[0]);
        for (int k = 1; k < R; k++) setpz2(y + (q_sRp + s*k), mulpz2(r[k], w[k]));
    }

    template <int R, int DIR> static inline ymm twiddleY(const int N, const int k, const int sp,
                                                        const_complex_vector W) noexcept
    {
        return duppz2(getpz(DIR < 0 ? W[k*sp] : W[N-k*sp]));
    }

    // Serial pass for an even s.
    template <int R, int DIR>
    static inline void corePassY(const int n, const int s,
                                 complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int N1 = N/R;
        const int m  = n/R;
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            ymm w[R];
            for (int k = 1; k < R; k++) w[k] = twiddleY<R, DIR>(N, k, sp, W);
            for (int q = 0; q < s; q += 2)
                coreY<R, DIR>(N1, s, q + sp, q + sp*R, w, x, y);
        }
    }

    // OpenMP pass for an even s: called by every thread of a parallel region.
    template <int R, int DIR>
    static inline void corePassYp(const int n, const int s,
                                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int N1 = N/R;
        const int ms = N/R/2;
#pragma omp for schedule(static)
        for (int h = 0; h < ms; h++) {
            const int i = 2*h;
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            ymm w[R];
            for (int k = 1; k < R; k++) w[k] = twiddleY<R, DIR>(N, k, sp, W);
            coreY<R, DIR>(N1, s, q + sp, q + sp*R, w, x, y);
        }
    }
#endif // __AVX__

    ///////////////////////////////////////////////////////////////////////////////
    // Forward Butterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    void fwdend2(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(zq+0, addpz2(a, b));
                setpz2(zq+s, subpz2(a, b));
            }
        }
        else {
            const xmm a = getpz(x[0]);
            const xmm b = getpz(x[1]);
            setpz(z[0], addpz(a, b));
            setpz(z[1], subpz(a, b));
        }
    }

    void fwdcore2(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/2;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/2;
        if (s >= 2) {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s2p = 2*sp;
                const ymm wp = duppz3(W[sp]);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s2p = y + q + s2p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    setpz2(yq_s2p+s*0,            addpz2(a, b));
                    setpz2(yq_s2p+s*1, mulpz2(wp, subpz2(a, b)));
                }
            }
        }
        else {
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_2p = y + 2*p;
                const xmm wp = getpz(W[p]);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                setpz(y_2p[0],           addpz(a, b));
                setpz(y_2p[1], mulpz(wp, subpz(a, b)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend4(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(zq+s*0, addpz2(apc,  bpd));
                setpz2(zq+s*1, subpz2(amc, jbmd));
                setpz2(zq+s*2, subpz2(apc,  bpd));
                setpz2(zq+s*3, addpz2(amc, jbmd));
            }
        }
        else {
            const xmm a = getpz(x[0]);
            const xmm b = getpz(x[1]);
            const xmm c = getpz(x[2]);
            const xmm d = getpz(x[3]);
            const xmm  apc =      addpz(a, c);
            const xmm  amc =      subpz(a, c);
            const xmm  bpd =      addpz(b, d);
            const xmm jbmd = jxpz(subpz(b, d));
            setpz(z[0], addpz(apc,  bpd));
            setpz(z[1], subpz(amc, jbmd));
            setpz(z[2], subpz(apc,  bpd));
            setpz(z[3], addpz(amc, jbmd));
        }
    }

    void fwdcore4(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/4;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/4;
        const int N2 = N1*2;
        const int N3 = N1*3;
        if (s >= 2) {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
                const ymm w1p = duppz3(W[1*sp]);
                const ymm w2p = duppz3(W[2*sp]);
                const ymm w3p = duppz3(W[3*sp]);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    const ymm c = getpz2(xq_sp+N2);
                    const ymm d = getpz2(xq_sp+N3);
                    const ymm  apc =       addpz2(a, c);
                    const ymm  amc =       subpz2(a, c);
                    const ymm  bpd =       addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                    setpz2(yq_s4p+s*1, mulpz2(w1p, subpz2(amc, jbmd)));
                    setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(yq_s4p+s*3, mulpz2(w3p, addpz2(amc, jbmd)));
                }
            }
        }
        else {
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                const xmm w1p = getpz(W[p]);
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                const xmm c = getpz(x_p[N2]);
                const xmm d = getpz(x_p[N3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(y_4p[0],            addpz(apc,  bpd));
                setpz(y_4p[1], mulpz(w1p, subpz(amc, jbmd)));
                setpz(y_4p[2], mulpz(w2p, subpz(apc,  bpd)));
                setpz(y_4p[3], mulpz(w3p, addpz(amc, jbmd)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend8(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm x0 = getpz2(xq+s*0);
                const ymm x1 = getpz2(xq+s*1);
                const ymm x2 = getpz2(xq+s*2);
                const ymm x3 = getpz2(xq+s*3);
                const ymm x4 = getpz2(xq+s*4);
                const ymm x5 = getpz2(xq+s*5);
                const ymm x6 = getpz2(xq+s*6);
                const ymm x7 = getpz2(xq+s*7);
                const ymm  a04 =       addpz2(x0, x4);
                const ymm  s04 =       subpz2(x0, x4);
                const ymm  a26 =       addpz2(x2, x6);
                const ymm js26 = jxpz2(subpz2(x2, x6));
                const ymm  a15 =       addpz2(x1, x5);
                const ymm  s15 =       subpz2(x1, x5);
                const ymm  a37 =       addpz2(x3, x7);
                const ymm js37 = jxpz2(subpz2(x3, x7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(zq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(zq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(zq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(zq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
            }
        }
        else {
            const xmm x0 = getpz(x[0]);
            const xmm x1 = getpz(x[1]);
            const xmm x2 = getpz(x[2]);
            const xmm x3 = getpz(x[3]);
            const xmm x4 = getpz(x[4]);
            const xmm x5 = getpz(x[5]);
            const xmm x6 = getpz(x[6]);
            const xmm x7 = getpz(x[7]);
            const xmm  a04 =      addpz(x0, x4);
            const xmm  s04 =      subpz(x0, x4);
            const xmm  a26 =      addpz(x2, x6);
            const xmm js26 = jxpz(subpz(x2, x6));
            const xmm  a15 =      addpz(x1, x5);
            const xmm  s15 =      subpz(x1, x5);
            const xmm  a37 =      addpz(x3, x7);
            const xmm js37 = jxpz(subpz(x3, x7));
            const xmm    a04_p1_a26 =       addpz(a04,  a26);
            const xmm    s04_mj_s26 =       subpz(s04, js26);
            const xmm    a04_m1_a26 =       subpz(a04,  a26);
            const xmm    s04_pj_s26 =       addpz(s04, js26);
            const xmm    a15_p1_a37 =       addpz(a15,  a37);
            const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
            const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
            const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
            setpz(z[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[1], addpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(z[2], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[3], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(z[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[5], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(z[6], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[7], addpz(s04_pj_s26, v8_s15_pj_s37));
        }
    }

    void fwdcore8(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/8;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/8;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const int N5 = N1*5;
        const int N6 = N1*6;
        const int N7 = N1*7;
        if (s >= 2) {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s8p = 8*sp;
                const ymm w1p = duppz3(W[1*sp]);
                const ymm w2p = duppz3(W[2*sp]);
                const ymm w3p = duppz3(W[3*sp]);
                const ymm w4p = mulpz2(w2p,w2p);
                const ymm w5p = mulpz2(w2p,w3p);
                const ymm w6p = mulpz2(w3p,w3p);
                const ymm w7p = mulpz2(w3p,w4p);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s8p = y + q + s8p;
                    const ymm x0 = getpz2(xq_sp+N0);
                    const ymm x1 = getpz2(xq_sp+N1);
                    const ymm x2 = getpz2(xq_sp+N2);
                    const ymm x3 = getpz2(xq_sp+N3);
                    const ymm x4 = getpz2(xq_sp+N4);
                    const ymm x5 = getpz2(xq_sp+N5);
                    const ymm x6 = getpz2(xq_sp+N6);
                    const ymm x7 = getpz2(xq_sp+N7);
                    const ymm  a04 =       addpz2(x0, x4);
                    const ymm  s04 =       subpz2(x0, x4);
                    const ymm  a26 =       addpz2(x2, x6);
                    const ymm js26 = jxpz2(subpz2(x2, x6));
                    const ymm  a15 =       addpz2(x1, x5);
                    const ymm  s15 =       subpz2(x1, x5);
                    const ymm  a37 =       addpz2(x3, x7);
                    const ymm js37 = jxpz2(subpz2(x3, x7));
                    const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                    const ymm    s04_mj_s26 =        subpz2(s04, js26);
                    const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                    const ymm    s04_pj_s26 =        addpz2(s04, js26);
                    const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                    const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                    const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                    const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                    setpz2(yq_s8p+s*0,             addpz2(a04_p1_a26,    a15_p1_a37));
                    setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
                    setpz2(yq_s8p+s*2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
                    setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
                    setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
                    setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
                    setpz2(yq_s8p+s*6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
                    setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
                }
            }
        }
        else {
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                const xmm w1p = getpz(W[p]);
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm w4p = mulpz(w2p,w2p);
                const xmm w5p = mulpz(w2p,w3p);
                const xmm w6p = mulpz(w3p,w3p);
                const xmm w7p = mulpz(w3p,w4p);
                const xmm x0 = getpz(x_p[N0]);
                const xmm x1 = getpz(x_p[N1]);
                const xmm x2 = getpz(x_p[N2]);
                const xmm x3 = getpz(x_p[N3]);
                const xmm x4 = getpz(x_p[N4]);
                const xmm x5 = getpz(x_p[N5]);
                const xmm x6 = getpz(x_p[N6]);
                const xmm x7 = getpz(x_p[N7]);
                const xmm  a04 =      addpz(x0, x4);
                const xmm  s04 =      subpz(x0, x4);
                const xmm  a26 =      addpz(x2, x6);
                const xmm js26 = jxpz(subpz(x2, x6));
                const xmm  a15 =      addpz(x1, x5);
                const xmm  s15 =      subpz(x1, x5);
                const xmm  a37 =      addpz(x3, x7);
                const xmm js37 = jxpz(subpz(x3, x7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                setpz(y_8p[0],            addpz(a04_p1_a26,    a15_p1_a37));
                setpz(y_8p[1], mulpz(w1p, addpz(s04_mj_s26, w8_s15_mj_s37)));
                setpz(y_8p[2], mulpz(w2p, subpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[3], mulpz(w3p, subpz(s04_pj_s26, v8_s15_pj_s37)));
                setpz(y_8p[4], mulpz(w4p, subpz(a04_p1_a26,    a15_p1_a37)));
                setpz(y_8p[5], mulpz(w5p, subpz(s04_mj_s26, w8_s15_mj_s37)));
                setpz(y_8p[6], mulpz(w6p, addpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[7], mulpz(w7p, addpz(s04_pj_s26, v8_s15_pj_s37)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend5(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            xmm r[5];
            bfly5<-1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]), getpz(x[q+s*4]), r);
            for (int k = 0; k < 5; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void fwdcore5(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassY<5, -1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/5;
        const int m  = n/5;
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[1*sp]);
            const cpx w2p = getpz(W[2*sp]);
            const cpx w3p = getpz(W[3*sp]);
            const cpx w4p = getpz(W[4*sp]);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                xmm r[5];
                bfly5<-1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]), getpz(x[q_sp+N1*4]), r);
                const int q_s5p = q + sp*5;
                setpz(y[q_s5p+s*0], r[0]);
                setpz(y[q_s5p+s*1], cpx(r[1])*w1p);
                setpz(y[q_s5p+s*2], cpx(r[2])*w2p);
                setpz(y[q_s5p+s*3], cpx(r[3])*w3p);
                setpz(y[q_s5p+s*4], cpx(r[4])*w4p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend3(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            xmm r[3];
            bfly3<-1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), r);
            for (int k = 0; k < 3; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void fwdcore3(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassY<3, -1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/3;
        const int m  = n/3;
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[1*sp]);
            const cpx w2p = getpz(W[2*sp]);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                xmm r[3];
                bfly3<-1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), r);
                const int q_s3p = q + sp*3;
                setpz(y[q_s3p+s*0], r[0]);
                setpz(y[q_s3p+s*1], cpx(r[1])*w1p);
                setpz(y[q_s3p+s*2], cpx(r[2])*w2p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Radix-7 butterfly kernel
    //
    // The 7-point DFT with the symmetry of the 7th roots of unity: with
    // t_j = x_j + x_(7-j) and d_j = x_j - x_(7-j) (j = 1, 2, 3),
    //   X_k     = x_0 + sum_j cos(2 pi j k / 7) t_j -+ i sum_j sin(2 pi j k / 7) d_j
    //   X_(7-k) = x_0 + sum_j cos(2 pi j k / 7) t_j +- i sum_j sin(2 pi j k / 7) d_j
    // (upper sign: forward, lower: inverse). That is 18 multiplications of a complex
    // number by a real constant instead of 36 complex multiplications.
    ///////////////////////////////////////////////////////////////////////////////

    // r[k] = k-th output; DIR = -1 forward, +1 inverse
    template <int DIR> static inline void bfly7(const xmm x0, const xmm x1, const xmm x2, const xmm x3,
                                                const xmm x4, const xmm x5, const xmm x6, xmm* r) noexcept force_inline3;
    template <int DIR> static inline void bfly7(const xmm x0, const xmm x1, const xmm x2, const xmm x3,
                                                const xmm x4, const xmm x5, const xmm x6, xmm* r) noexcept
    {
        const xmm c1 = cmplx(R7_C1, R7_C1);
        const xmm c2 = cmplx(R7_C2, R7_C2);
        const xmm c3 = cmplx(R7_C3, R7_C3);
        const xmm s1 = cmplx(R7_S1, R7_S1);
        const xmm s2 = cmplx(R7_S2, R7_S2);
        const xmm s3 = cmplx(R7_S3, R7_S3);
        const xmm t1 = addpz(x1, x6), d1 = subpz(x1, x6);
        const xmm t2 = addpz(x2, x5), d2 = subpz(x2, x5);
        const xmm t3 = addpz(x3, x4), d3 = subpz(x3, x4);
        r[0] = addpz(x0, addpz(t1, addpz(t2, t3)));
        const xmm a1 = addpz(x0, addpz(mulpd(c1, t1), addpz(mulpd(c2, t2), mulpd(c3, t3))));
        const xmm a2 = addpz(x0, addpz(mulpd(c2, t1), addpz(mulpd(c3, t2), mulpd(c1, t3))));
        const xmm a3 = addpz(x0, addpz(mulpd(c3, t1), addpz(mulpd(c1, t2), mulpd(c2, t3))));
        const xmm b1 = addpz(mulpd(s1, d1), addpz(mulpd(s2, d2), mulpd(s3, d3)));
        const xmm b2 = subpz(subpz(mulpd(s2, d1), mulpd(s3, d2)), mulpd(s1, d3));
        const xmm b3 = addpz(subpz(mulpd(s3, d1), mulpd(s1, d2)), mulpd(s2, d3));
        if (DIR < 0) {
            r[1] = addpz(a1, mjxpz(b1)); r[6] = addpz(a1, jxpz(b1));
            r[2] = addpz(a2, mjxpz(b2)); r[5] = addpz(a2, jxpz(b2));
            r[3] = addpz(a3, mjxpz(b3)); r[4] = addpz(a3, jxpz(b3));
        }
        else {
            r[1] = addpz(a1, jxpz(b1)); r[6] = addpz(a1, mjxpz(b1));
            r[2] = addpz(a2, jxpz(b2)); r[5] = addpz(a2, mjxpz(b2));
            r[3] = addpz(a3, jxpz(b3)); r[4] = addpz(a3, mjxpz(b3));
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void fwdend7(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            xmm r[7];
            bfly7<-1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]),
                      getpz(x[q+s*4]), getpz(x[q+s*5]), getpz(x[q+s*6]), r);
            for (int k = 0; k < 7; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void fwdcore7(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassY<7, -1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int m  = n/7;
        const int N1 = N/7;
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[1*sp]);
            const cpx w2p = getpz(W[2*sp]);
            const cpx w3p = getpz(W[3*sp]);
            const cpx w4p = getpz(W[4*sp]);
            const cpx w5p = getpz(W[5*sp]);
            const cpx w6p = getpz(W[6*sp]);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                xmm r[7];
                bfly7<-1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]),
                          getpz(x[q_sp+N1*4]), getpz(x[q_sp+N1*5]), getpz(x[q_sp+N1*6]), r);
                const int q_s7p = q + sp*7;
                setpz(y[q_s7p+s*0], r[0]);
                setpz(y[q_s7p+s*1], cpx(r[1])*w1p);
                setpz(y[q_s7p+s*2], cpx(r[2])*w2p);
                setpz(y[q_s7p+s*3], cpx(r[3])*w3p);
                setpz(y[q_s7p+s*4], cpx(r[4])*w4p);
                setpz(y[q_s7p+s*5], cpx(r[5])*w5p);
                setpz(y[q_s7p+s*6], cpx(r[6])*w6p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size FFT except Radix-2,3,5,7
    ///////////////////////////////////////////////////////////////////////////////

    void fwdfftany(const int r, const int n, const int s, const bool eo,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        constexpr xmm zero = { 0, 0 };
        const int N = n*s;
        int k = r;
        while (n%k != 0) {
            if (k*k > n) { k = n; break; }
            k += 2;
        }
        if (k == n) {
            for (int q = 0; q < s; q++) {
                for (int i = 0; i < k; i++) {
                    cpx z = zero;
                    int ij = 0;
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[s*ij]);
                        z = z + a*wij;
                        ij += i; if (ij >= k) ij -= k;
                    }
                    setpz(y[q+s*i], z);
                }
            }
            if (!eo) for (int p = 0; p < N; p++) setpz(x[p], getpz(y[p]));
        }
        else {
            const int m  = n/k;
            const int ms = m*s;
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                for (int q = 0; q < s; q++) {
                    const int q_sp  = q + sp;
                    const int q_spk = q + sp*k;
                    for (int i = 0; i < k; i++) {
                        cpx z = zero;
                        int ij = 0;
                        for (int j = 0; j < k; j++) {
                            const cpx a   = getpz(x[q_sp+ms*j]);
                            const cpx wij = getpz(W[ms*ij]);
                            z = z + a*wij;
                            ij += i; if (ij >= k) ij -= k;
                        }
                        const cpx wip = getpz(W[i*sp]);
                        setpz(y[q_spk+s*i], z * wip);
                    }
                }
            }
            fwdfftany(k, m, k*s, !eo, y, x, W);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Mixed Radix FFT
    ///////////////////////////////////////////////////////////////////////////////

    void fwdfft(const int n, const int s, const bool eo,
                complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N = n*s;
        if (N < 2) return;
        if (n%8 == 0) {
            if (n == 8)
                fwdend8(s, eo, x, y);
            else {
                fwdcore8(n, s, x, y, W);
                fwdfft(n/8, 8*s, !eo, y, x, W);
            }
        }
        else if (n%4 == 0) {
            if (n == 4)
                fwdend4(s, eo, x, y);
            else {
                fwdcore4(n, s, x, y, W);
                fwdfft(n/4, 4*s, !eo, y, x, W);
            }
        }
        else if (n%2 == 0) {
            if (n == 2)
                fwdend2(s, eo, x, y);
            else {
                fwdcore2(n, s, x, y, W);
                fwdfft(n/2, 2*s, !eo, y, x, W);
            }
        }
        else if (n%5 == 0) {
            if (n == 5)
                fwdend5(s, eo, x, y);
            else {
                fwdcore5(n, s, x, y, W);
                fwdfft(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                fwdend3(s, eo, x, y);
            else {
                fwdcore3(n, s, x, y, W);
                fwdfft(n/3, 3*s, !eo, y, x, W);
            }
        }
        else if (n%7 == 0) {
            if (n == 7)
                fwdend7(s, eo, x, y);
            else {
                fwdcore7(n, s, x, y, W);
                fwdfft(n/7, 7*s, !eo, y, x, W);
            }
        }
        else fwdfftany(7, n, s, eo, x, y, W);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Inverse Butterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    void invend2(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(zq+0, addpz2(a, b));
                setpz2(zq+s, subpz2(a, b));
            }
        }
        else {
            const xmm a = getpz(x[0]);
            const xmm b = getpz(x[1]);
            setpz(z[0], addpz(a, b));
            setpz(z[1], subpz(a, b));
        }
    }

    void invcore2(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/2;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/2;
        if (s >= 2) {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s2p = 2*sp;
                const ymm wp = duppz3(W[N-sp]);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s2p = y + q + s2p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    setpz2(yq_s2p+s*0,            addpz2(a, b));
                    setpz2(yq_s2p+s*1, mulpz2(wp, subpz2(a, b)));
                }
            }
        }
        else {
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_2p = y + 2*p;
                const xmm wp = getpz(W[N-p]);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                setpz(y_2p[0],           addpz(a, b));
                setpz(y_2p[1], mulpz(wp, subpz(a, b)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend4(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(zq+s*0, addpz2(apc,  bpd));
                setpz2(zq+s*1, addpz2(amc, jbmd));
                setpz2(zq+s*2, subpz2(apc,  bpd));
                setpz2(zq+s*3, subpz2(amc, jbmd));
            }
        }
        else {
            const xmm a = getpz(x[0]);
            const xmm b = getpz(x[1]);
            const xmm c = getpz(x[2]);
            const xmm d = getpz(x[3]);
            const xmm  apc =      addpz(a, c);
            const xmm  amc =      subpz(a, c);
            const xmm  bpd =      addpz(b, d);
            const xmm jbmd = jxpz(subpz(b, d));
            setpz(z[0], addpz(apc,  bpd));
            setpz(z[1], addpz(amc, jbmd));
            setpz(z[2], subpz(apc,  bpd));
            setpz(z[3], subpz(amc, jbmd));
        }
    }

    void invcore4(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/4;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/4;
        const int N2 = N1*2;
        const int N3 = N1*3;
        if (s >= 2) {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
                const ymm w1p = duppz3(W[N-1*sp]);
                const ymm w2p = duppz3(W[N-2*sp]);
                const ymm w3p = duppz3(W[N-3*sp]);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    const ymm c = getpz2(xq_sp+N2);
                    const ymm d = getpz2(xq_sp+N3);
                    const ymm  apc =       addpz2(a, c);
                    const ymm  amc =       subpz2(a, c);
                    const ymm  bpd =       addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                    setpz2(yq_s4p+s*1, mulpz2(w1p, addpz2(amc, jbmd)));
                    setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(yq_s4p+s*3, mulpz2(w3p, subpz2(amc, jbmd)));
                }
            }
        }
        else {
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                const xmm w1p = cnjpz(getpz(W[p]));
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                const xmm c = getpz(x_p[N2]);
                const xmm d = getpz(x_p[N3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(y_4p[0],            addpz(apc,  bpd));
                setpz(y_4p[1], mulpz(w1p, addpz(amc, jbmd)));
                setpz(y_4p[2], mulpz(w2p, subpz(apc,  bpd)));
                setpz(y_4p[3], mulpz(w3p, subpz(amc, jbmd)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend8(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm x0 = getpz2(xq+s*0);
                const ymm x1 = getpz2(xq+s*1);
                const ymm x2 = getpz2(xq+s*2);
                const ymm x3 = getpz2(xq+s*3);
                const ymm x4 = getpz2(xq+s*4);
                const ymm x5 = getpz2(xq+s*5);
                const ymm x6 = getpz2(xq+s*6);
                const ymm x7 = getpz2(xq+s*7);
                const ymm  a04 =       addpz2(x0, x4);
                const ymm  s04 =       subpz2(x0, x4);
                const ymm  a26 =       addpz2(x2, x6);
                const ymm js26 = jxpz2(subpz2(x2, x6));
                const ymm  a15 =       addpz2(x1, x5);
                const ymm  s15 =       subpz2(x1, x5);
                const ymm  a37 =       addpz2(x3, x7);
                const ymm js37 = jxpz2(subpz2(x3, x7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(zq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(zq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(zq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(zq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
            }
        }
        else {
            const xmm x0 = getpz(x[0]);
            const xmm x1 = getpz(x[1]);
            const xmm x2 = getpz(x[2]);
            const xmm x3 = getpz(x[3]);
            const xmm x4 = getpz(x[4]);
            const xmm x5 = getpz(x[5]);
            const xmm x6 = getpz(x[6]);
            const xmm x7 = getpz(x[7]);
            const xmm  a04 =      addpz(x0, x4);
            const xmm  s04 =      subpz(x0, x4);
            const xmm  a26 =      addpz(x2, x6);
            const xmm js26 = jxpz(subpz(x2, x6));
            const xmm  a15 =      addpz(x1, x5);
            const xmm  s15 =      subpz(x1, x5);
            const xmm  a37 =      addpz(x3, x7);
            const xmm js37 = jxpz(subpz(x3, x7));
            const xmm    a04_p1_a26 =       addpz(a04,  a26);
            const xmm    s04_pj_s26 =       addpz(s04, js26);
            const xmm    a04_m1_a26 =       subpz(a04,  a26);
            const xmm    s04_mj_s26 =       subpz(s04, js26);
            const xmm    a15_p1_a37 =       addpz(a15,  a37);
            const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
            const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
            const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
            setpz(z[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[1], addpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(z[2], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[3], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(z[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(z[5], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(z[6], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(z[7], addpz(s04_mj_s26, w8_s15_mj_s37));
        }
    }

    void invcore8(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/8;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/8;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const int N5 = N1*5;
        const int N6 = N1*6;
        const int N7 = N1*7;
        if (s >= 2) {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s8p = 8*sp;
                const ymm w1p = duppz3(W[N-1*sp]);
                const ymm w2p = duppz3(W[N-2*sp]);
                const ymm w3p = duppz3(W[N-3*sp]);
                const ymm w4p = mulpz2(w2p,w2p);
                const ymm w5p = mulpz2(w2p,w3p);
                const ymm w6p = mulpz2(w3p,w3p);
                const ymm w7p = mulpz2(w3p,w4p);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s8p = y + q + s8p;
                    const ymm x0 = getpz2(xq_sp+N0);
                    const ymm x1 = getpz2(xq_sp+N1);
                    const ymm x2 = getpz2(xq_sp+N2);
                    const ymm x3 = getpz2(xq_sp+N3);
                    const ymm x4 = getpz2(xq_sp+N4);
                    const ymm x5 = getpz2(xq_sp+N5);
                    const ymm x6 = getpz2(xq_sp+N6);
                    const ymm x7 = getpz2(xq_sp+N7);
                    const ymm  a04 =       addpz2(x0, x4);
                    const ymm  s04 =       subpz2(x0, x4);
                    const ymm  a26 =       addpz2(x2, x6);
                    const ymm js26 = jxpz2(subpz2(x2, x6));
                    const ymm  a15 =       addpz2(x1, x5);
                    const ymm  s15 =       subpz2(x1, x5);
                    const ymm  a37 =       addpz2(x3, x7);
                    const ymm js37 = jxpz2(subpz2(x3, x7));
                    const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                    const ymm    s04_pj_s26 =        addpz2(s04, js26);
                    const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                    const ymm    s04_mj_s26 =        subpz2(s04, js26);
                    const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                    const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                    const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                    const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                    setpz2(yq_s8p+s*0,             addpz2(a04_p1_a26,    a15_p1_a37));
                    setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
                    setpz2(yq_s8p+s*2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
                    setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
                    setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
                    setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
                    setpz2(yq_s8p+s*6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
                    setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
                }
            }
        }
        else {
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                const xmm w1p = cnjpz(getpz(W[p]));
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm w4p = mulpz(w2p,w2p);
                const xmm w5p = mulpz(w2p,w3p);
                const xmm w6p = mulpz(w3p,w3p);
                const xmm w7p = mulpz(w3p,w4p);
                const xmm x0 = getpz(x_p[N0]);
                const xmm x1 = getpz(x_p[N1]);
                const xmm x2 = getpz(x_p[N2]);
                const xmm x3 = getpz(x_p[N3]);
                const xmm x4 = getpz(x_p[N4]);
                const xmm x5 = getpz(x_p[N5]);
                const xmm x6 = getpz(x_p[N6]);
                const xmm x7 = getpz(x_p[N7]);
                const xmm  a04 =      addpz(x0, x4);
                const xmm  s04 =      subpz(x0, x4);
                const xmm  a26 =      addpz(x2, x6);
                const xmm js26 = jxpz(subpz(x2, x6));
                const xmm  a15 =      addpz(x1, x5);
                const xmm  s15 =      subpz(x1, x5);
                const xmm  a37 =      addpz(x3, x7);
                const xmm js37 = jxpz(subpz(x3, x7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                setpz(y_8p[0],            addpz(a04_p1_a26,    a15_p1_a37));
                setpz(y_8p[1], mulpz(w1p, addpz(s04_pj_s26, v8_s15_pj_s37)));
                setpz(y_8p[2], mulpz(w2p, addpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[3], mulpz(w3p, subpz(s04_mj_s26, w8_s15_mj_s37)));
                setpz(y_8p[4], mulpz(w4p, subpz(a04_p1_a26,    a15_p1_a37)));
                setpz(y_8p[5], mulpz(w5p, subpz(s04_pj_s26, v8_s15_pj_s37)));
                setpz(y_8p[6], mulpz(w6p, subpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[7], mulpz(w7p, addpz(s04_mj_s26, w8_s15_mj_s37)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend5(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            xmm r[5];
            bfly5<1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]), getpz(x[q+s*4]), r);
            for (int k = 0; k < 5; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void invcore5(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassY<5, 1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/5;
        const int m  = n/5;
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[N-1*sp]);
            const cpx w2p = getpz(W[N-2*sp]);
            const cpx w3p = getpz(W[N-3*sp]);
            const cpx w4p = getpz(W[N-4*sp]);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                xmm r[5];
                bfly5<1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]), getpz(x[q_sp+N1*4]), r);
                const int q_s5p = q + sp*5;
                setpz(y[q_s5p+s*0], r[0]);
                setpz(y[q_s5p+s*1], cpx(r[1])*w1p);
                setpz(y[q_s5p+s*2], cpx(r[2])*w2p);
                setpz(y[q_s5p+s*3], cpx(r[3])*w3p);
                setpz(y[q_s5p+s*4], cpx(r[4])*w4p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend3(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            xmm r[3];
            bfly3<1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), r);
            for (int k = 0; k < 3; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void invcore3(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassY<3, 1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/3;
        const int m  = n/3;
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[N-1*sp]);
            const cpx w2p = getpz(W[N-2*sp]);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                xmm r[3];
                bfly3<1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), r);
                const int q_s3p = q + sp*3;
                setpz(y[q_s3p+s*0], r[0]);
                setpz(y[q_s3p+s*1], cpx(r[1])*w1p);
                setpz(y[q_s3p+s*2], cpx(r[2])*w2p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void invend7(const int s, const bool eo,
                 complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            xmm r[7];
            bfly7<1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]),
                      getpz(x[q+s*4]), getpz(x[q+s*5]), getpz(x[q+s*6]), r);
            for (int k = 0; k < 7; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void invcore7(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassY<7, 1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int m  = n/7;
        const int N1 = N/7;
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[N-1*sp]);
            const cpx w2p = getpz(W[N-2*sp]);
            const cpx w3p = getpz(W[N-3*sp]);
            const cpx w4p = getpz(W[N-4*sp]);
            const cpx w5p = getpz(W[N-5*sp]);
            const cpx w6p = getpz(W[N-6*sp]);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                xmm r[7];
                bfly7<1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]),
                          getpz(x[q_sp+N1*4]), getpz(x[q_sp+N1*5]), getpz(x[q_sp+N1*6]), r);
                const int q_s7p = q + sp*7;
                setpz(y[q_s7p+s*0], r[0]);
                setpz(y[q_s7p+s*1], cpx(r[1])*w1p);
                setpz(y[q_s7p+s*2], cpx(r[2])*w2p);
                setpz(y[q_s7p+s*3], cpx(r[3])*w3p);
                setpz(y[q_s7p+s*4], cpx(r[4])*w4p);
                setpz(y[q_s7p+s*5], cpx(r[5])*w5p);
                setpz(y[q_s7p+s*6], cpx(r[6])*w6p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size IFFT except Radix-2,3,5,7
    ///////////////////////////////////////////////////////////////////////////////

    void invfftany(const int r, const int n, const int s, const bool eo,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        constexpr xmm zero = { 0, 0 };
        const int N = n*s;
        int k = r;
        while (n%k != 0) {
            if (k*k > n) { k = n; break; }
            k += 2;
        }
        if (k == n) {
            for (int q = 0; q < s; q++) {
                for (int i = 0; i < k; i++) {
                    cpx z = zero;
                    int ij = 0;
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[N-s*ij]);
                        z = z + a*wij;
                        ij += i; if (ij >= k) ij -= k;
                    }
                    setpz(y[q+s*i], z);
                }
            }
            if (!eo) for (int p = 0; p < N; p++) setpz(x[p], getpz(y[p]));
        }
        else {
            const int m  = n/k;
            const int ms = m*s;
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                for (int q = 0; q < s; q++) {
                    const int q_sp  = q + sp;
                    const int q_spk = q + sp*k;
                    for (int i = 0; i < k; i++) {
                        cpx z = zero;
                        int ij = 0;
                        for (int j = 0; j < k; j++) {
                            const cpx a   = getpz(x[q_sp+ms*j]);
                            const cpx wij = getpz(W[N-ms*ij]);
                            z = z + a*wij;
                            ij += i; if (ij >= k) ij -= k;
                        }
                        const cpx wip = getpz(W[N-i*sp]);
                        setpz(y[q_spk+s*i], z * wip);
                    }
                }
            }
            invfftany(k, m, k*s, !eo, y, x, W);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Mixed Radix IFFT
    ///////////////////////////////////////////////////////////////////////////////

    void invfft(const int n, const int s, const bool eo,
                complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N = n*s;
        if (N < 2) return;
        if (n%8 == 0) {
            if (n == 8)
                invend8(s, eo, x, y);
            else {
                invcore8(n, s, x, y, W);
                invfft(n/8, 8*s, !eo, y, x, W);
            }
        }
        else if (n%4 == 0) {
            if (n == 4)
                invend4(s, eo, x, y);
            else {
                invcore4(n, s, x, y, W);
                invfft(n/4, 4*s, !eo, y, x, W);
            }
        }
        else if (n%2 == 0) {
            if (n == 2)
                invend2(s, eo, x, y);
            else {
                invcore2(n, s, x, y, W);
                invfft(n/2, 2*s, !eo, y, x, W);
            }
        }
        else if (n%5 == 0) {
            if (n == 5)
                invend5(s, eo, x, y);
            else {
                invcore5(n, s, x, y, W);
                invfft(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                invend3(s, eo, x, y);
            else {
                invcore3(n, s, x, y, W);
                invfft(n/3, 3*s, !eo, y, x, W);
            }
        }
        else if (n%7 == 0) {
            if (n == 7)
                invend7(s, eo, x, y);
            else {
                invcore7(n, s, x, y, W);
                invfft(n/7, 7*s, !eo, y, x, W);
            }
        }
        else invfftany(7, n, s, eo, x, y, W);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Forward Butterfly Operation with OpenMP
    ///////////////////////////////////////////////////////////////////////////////

    void fwdend2p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(zq+0, addpz2(a, b));
                setpz2(zq+s, subpz2(a, b));
            }
        }
        else {
#pragma omp single
            {
                const xmm a = getpz(x[0]);
                const xmm b = getpz(x[1]);
                setpz(z[0], addpz(a, b));
                setpz(z[1], subpz(a, b));
            }
        }
    }

    void fwdcore2p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/2;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/2;
        if (s >= 2) {
#pragma omp for schedule(static)
            for (int i = 0; i < N/4; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s2p = 2*sp;
                const ymm wp = duppz3(W[sp]);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s2p = y + q + s2p;
                const ymm a = getpz2(xq_sp+N0);
                const ymm b = getpz2(xq_sp+N1);
                setpz2(yq_s2p+s*0,            addpz2(a, b));
                setpz2(yq_s2p+s*1, mulpz2(wp, subpz2(a, b)));
            }
        }
        else {
#pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_2p = y + 2*p;
                const xmm wp = getpz(W[p]);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                setpz(y_2p[0],           addpz(a, b));
                setpz(y_2p[1], mulpz(wp, subpz(a, b)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend4p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(zq+s*0, addpz2(apc,  bpd));
                setpz2(zq+s*1, subpz2(amc, jbmd));
                setpz2(zq+s*2, subpz2(apc,  bpd));
                setpz2(zq+s*3, addpz2(amc, jbmd));
            }
        }
        else {
#pragma omp single
            {
                const xmm a = getpz(x[0]);
                const xmm b = getpz(x[1]);
                const xmm c = getpz(x[2]);
                const xmm d = getpz(x[3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(z[0], addpz(apc,  bpd));
                setpz(z[1], subpz(amc, jbmd));
                setpz(z[2], subpz(apc,  bpd));
                setpz(z[3], addpz(amc, jbmd));
            }
        }
    }

    void fwdcore4p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/4;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/4;
        const int N2 = N1*2;
        const int N3 = N1*3;
        if (s >= 2) {
#pragma omp for schedule(static)
            for (int i = 0; i < N/8; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s4p = 4*sp;
                const ymm w1p = duppz3(W[1*sp]);
                const ymm w2p = duppz3(W[2*sp]);
                const ymm w3p = duppz3(W[3*sp]);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s4p = y + q + s4p;
                const ymm a = getpz2(xq_sp+N0);
                const ymm b = getpz2(xq_sp+N1);
                const ymm c = getpz2(xq_sp+N2);
                const ymm d = getpz2(xq_sp+N3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                setpz2(yq_s4p+s*1, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(yq_s4p+s*3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
        }
        else {
#pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                const xmm w1p = getpz(W[p]);
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                const xmm c = getpz(x_p[N2]);
                const xmm d = getpz(x_p[N3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(y_4p[0],            addpz(apc,  bpd));
                setpz(y_4p[1], mulpz(w1p, subpz(amc, jbmd)));
                setpz(y_4p[2], mulpz(w2p, subpz(apc,  bpd)));
                setpz(y_4p[3], mulpz(w3p, addpz(amc, jbmd)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend8p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm x0 = getpz2(xq+s*0);
                const ymm x1 = getpz2(xq+s*1);
                const ymm x2 = getpz2(xq+s*2);
                const ymm x3 = getpz2(xq+s*3);
                const ymm x4 = getpz2(xq+s*4);
                const ymm x5 = getpz2(xq+s*5);
                const ymm x6 = getpz2(xq+s*6);
                const ymm x7 = getpz2(xq+s*7);
                const ymm  a04 =       addpz2(x0, x4);
                const ymm  s04 =       subpz2(x0, x4);
                const ymm  a26 =       addpz2(x2, x6);
                const ymm js26 = jxpz2(subpz2(x2, x6));
                const ymm  a15 =       addpz2(x1, x5);
                const ymm  s15 =       subpz2(x1, x5);
                const ymm  a37 =       addpz2(x3, x7);
                const ymm js37 = jxpz2(subpz2(x3, x7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(zq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(zq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(zq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(zq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
            }
        }
        else {
#pragma omp single
            {
                const xmm x0 = getpz(x[0]);
                const xmm x1 = getpz(x[1]);
                const xmm x2 = getpz(x[2]);
                const xmm x3 = getpz(x[3]);
                const xmm x4 = getpz(x[4]);
                const xmm x5 = getpz(x[5]);
                const xmm x6 = getpz(x[6]);
                const xmm x7 = getpz(x[7]);
                const xmm  a04 =      addpz(x0, x4);
                const xmm  s04 =      subpz(x0, x4);
                const xmm  a26 =      addpz(x2, x6);
                const xmm js26 = jxpz(subpz(x2, x6));
                const xmm  a15 =      addpz(x1, x5);
                const xmm  s15 =      subpz(x1, x5);
                const xmm  a37 =      addpz(x3, x7);
                const xmm js37 = jxpz(subpz(x3, x7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                setpz(z[0], addpz(a04_p1_a26,    a15_p1_a37));
                setpz(z[1], addpz(s04_mj_s26, w8_s15_mj_s37));
                setpz(z[2], subpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(z[3], subpz(s04_pj_s26, v8_s15_pj_s37));
                setpz(z[4], subpz(a04_p1_a26,    a15_p1_a37));
                setpz(z[5], subpz(s04_mj_s26, w8_s15_mj_s37));
                setpz(z[6], addpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(z[7], addpz(s04_pj_s26, v8_s15_pj_s37));
            }
        }
    }

    void fwdcore8p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/8;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/8;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const int N5 = N1*5;
        const int N6 = N1*6;
        const int N7 = N1*7;
        if (s >= 2) {
#pragma omp for schedule(static)
            for (int i = 0; i < N/16; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s8p = 8*sp;
                const ymm w1p = duppz3(W[1*sp]);
                const ymm w2p = duppz3(W[2*sp]);
                const ymm w3p = duppz3(W[3*sp]);
                const ymm w4p = mulpz2(w2p,w2p);
                const ymm w5p = mulpz2(w2p,w3p);
                const ymm w6p = mulpz2(w3p,w3p);
                const ymm w7p = mulpz2(w3p,w4p);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s8p = y + q + s8p;
                const ymm x0 = getpz2(xq_sp+N0);
                const ymm x1 = getpz2(xq_sp+N1);
                const ymm x2 = getpz2(xq_sp+N2);
                const ymm x3 = getpz2(xq_sp+N3);
                const ymm x4 = getpz2(xq_sp+N4);
                const ymm x5 = getpz2(xq_sp+N5);
                const ymm x6 = getpz2(xq_sp+N6);
                const ymm x7 = getpz2(xq_sp+N7);
                const ymm  a04 =       addpz2(x0, x4);
                const ymm  s04 =       subpz2(x0, x4);
                const ymm  a26 =       addpz2(x2, x6);
                const ymm js26 = jxpz2(subpz2(x2, x6));
                const ymm  a15 =       addpz2(x1, x5);
                const ymm  s15 =       subpz2(x1, x5);
                const ymm  a37 =       addpz2(x3, x7);
                const ymm js37 = jxpz2(subpz2(x3, x7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(yq_s8p+s*0,             addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
                setpz2(yq_s8p+s*2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
                setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
                setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
                setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
                setpz2(yq_s8p+s*6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
                setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
            }
        }
        else {
#pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                const xmm w1p = getpz(W[p]);
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm w4p = mulpz(w2p,w2p);
                const xmm w5p = mulpz(w2p,w3p);
                const xmm w6p = mulpz(w3p,w3p);
                const xmm w7p = mulpz(w3p,w4p);
                const xmm x0 = getpz(x_p[N0]);
                const xmm x1 = getpz(x_p[N1]);
                const xmm x2 = getpz(x_p[N2]);
                const xmm x3 = getpz(x_p[N3]);
                const xmm x4 = getpz(x_p[N4]);
                const xmm x5 = getpz(x_p[N5]);
                const xmm x6 = getpz(x_p[N6]);
                const xmm x7 = getpz(x_p[N7]);
                const xmm  a04 =      addpz(x0, x4);
                const xmm  s04 =      subpz(x0, x4);
                const xmm  a26 =      addpz(x2, x6);
                const xmm js26 = jxpz(subpz(x2, x6));
                const xmm  a15 =      addpz(x1, x5);
                const xmm  s15 =      subpz(x1, x5);
                const xmm  a37 =      addpz(x3, x7);
                const xmm js37 = jxpz(subpz(x3, x7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                setpz(y_8p[0],            addpz(a04_p1_a26,    a15_p1_a37));
                setpz(y_8p[1], mulpz(w1p, addpz(s04_mj_s26, w8_s15_mj_s37)));
                setpz(y_8p[2], mulpz(w2p, subpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[3], mulpz(w3p, subpz(s04_pj_s26, v8_s15_pj_s37)));
                setpz(y_8p[4], mulpz(w4p, subpz(a04_p1_a26,    a15_p1_a37)));
                setpz(y_8p[5], mulpz(w5p, subpz(s04_mj_s26, w8_s15_mj_s37)));
                setpz(y_8p[6], mulpz(w6p, addpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[7], mulpz(w7p, addpz(s04_pj_s26, v8_s15_pj_s37)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend5p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            xmm r[5];
            bfly5<-1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]), getpz(x[q+s*4]), r);
            for (int k = 0; k < 5; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void fwdcore5p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassYp<5, -1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/5;
        const int ms = N/5;
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[1*sp]);
            const cpx w2p = getpz(W[2*sp]);
            const cpx w3p = getpz(W[3*sp]);
            const cpx w4p = getpz(W[4*sp]);
            const int q_sp = q + sp;
            xmm r[5];
            bfly5<-1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]), getpz(x[q_sp+N1*4]), r);
            const int q_s5p = q + sp*5;
            setpz(y[q_s5p+s*0], r[0]);
            setpz(y[q_s5p+s*1], cpx(r[1])*w1p);
            setpz(y[q_s5p+s*2], cpx(r[2])*w2p);
            setpz(y[q_s5p+s*3], cpx(r[3])*w3p);
            setpz(y[q_s5p+s*4], cpx(r[4])*w4p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend3p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            xmm r[3];
            bfly3<-1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), r);
            for (int k = 0; k < 3; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void fwdcore3p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassYp<3, -1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/3;
        const int ms = N/3;
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[1*sp]);
            const cpx w2p = getpz(W[2*sp]);
            const int q_sp = q + sp;
            xmm r[3];
            bfly3<-1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), r);
            const int q_s3p = q + sp*3;
            setpz(y[q_s3p+s*0], r[0]);
            setpz(y[q_s3p+s*1], cpx(r[1])*w1p);
            setpz(y[q_s3p+s*2], cpx(r[2])*w2p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void fwdend7p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            xmm r[7];
            bfly7<-1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]),
                      getpz(x[q+s*4]), getpz(x[q+s*5]), getpz(x[q+s*6]), r);
            for (int k = 0; k < 7; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void fwdcore7p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassYp<7, -1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int ms = N/7;
        const int N1 = N/7;
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[1*sp]);
            const cpx w2p = getpz(W[2*sp]);
            const cpx w3p = getpz(W[3*sp]);
            const cpx w4p = getpz(W[4*sp]);
            const cpx w5p = getpz(W[5*sp]);
            const cpx w6p = getpz(W[6*sp]);
            const int q_sp = q + sp;
            xmm r[7];
            bfly7<-1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]),
                      getpz(x[q_sp+N1*4]), getpz(x[q_sp+N1*5]), getpz(x[q_sp+N1*6]), r);
            const int q_s7p = q + sp*7;
            setpz(y[q_s7p+s*0], r[0]);
            setpz(y[q_s7p+s*1], cpx(r[1])*w1p);
            setpz(y[q_s7p+s*2], cpx(r[2])*w2p);
            setpz(y[q_s7p+s*3], cpx(r[3])*w3p);
            setpz(y[q_s7p+s*4], cpx(r[4])*w4p);
            setpz(y[q_s7p+s*5], cpx(r[5])*w5p);
            setpz(y[q_s7p+s*6], cpx(r[6])*w6p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size FFT except Radix-2,3,5,7 with OpenMP
    ///////////////////////////////////////////////////////////////////////////////

    void fwdfftanyp(const int r, const int n, const int s, const bool eo,
                    complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        constexpr xmm zero = { 0, 0 };
        const int N = n*s;
        int k = r;
        while (n%k != 0) {
            if (k*k > n) { k = n; break; }
            k += 2;
        }
        if (k == n) {
#pragma omp for schedule(static)
            for (int q = 0; q < s; q++) {
                for (int i = 0; i < k; i++) {
                    cpx z = zero;
                    int ij = 0;
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[s*ij]);
                        z = z + a*wij;
                        ij += i; if (ij >= k) ij -= k;
                    }
                    setpz(y[q+s*i], z);
                }
            }
            if (!eo) {
#pragma omp for schedule(static) nowait
                for (int p = 0; p < N; p++) setpz(x[p], getpz(y[p]));
            }
        }
        else {
            const int m  = n/k;
            const int ms = m*s;
#pragma omp for schedule(static)
            for (int h = 0; h < ms; h++) {
                const int p = h / s;
                const int q = h % s;
                const int sp = s*p;
                const int q_sp  = q + sp;
                const int q_spk = q + sp*k;
                for (int i = 0; i < k; i++) {
                    cpx z = zero;
                    int ij = 0;
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q_sp+ms*j]);
                        const cpx wij = getpz(W[ms*ij]);
                        z = z + a*wij;
                        ij += i; if (ij >= k) ij -= k;
                    }
                    const cpx wip = getpz(W[i*sp]);
                    setpz(y[q_spk+s*i], z * wip);
                }
            }
            fwdfftanyp(k, m, k*s, !eo, y, x, W);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Mixed Radix FFT with OpenMP
    ///////////////////////////////////////////////////////////////////////////////

    void fwdfftp(const int n, const int s, const bool eo,
                 complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N = n*s;
        if (N < 2) return;
        if (n%8 == 0) {
            if (n == 8)
                fwdend8p(s, eo, x, y);
            else {
                fwdcore8p(n, s, x, y, W);
                fwdfftp(n/8, 8*s, !eo, y, x, W);
            }
        }
        else if (n%4 == 0) {
            if (n == 4)
                fwdend4p(s, eo, x, y);
            else {
                fwdcore4p(n, s, x, y, W);
                fwdfftp(n/4, 4*s, !eo, y, x, W);
            }
        }
        else if (n%2 == 0) {
            if (n == 2)
                fwdend2p(s, eo, x, y);
            else {
                fwdcore2p(n, s, x, y, W);
                fwdfftp(n/2, 2*s, !eo, y, x, W);
            }
        }
        else if (n%5 == 0) {
            if (n == 5)
                fwdend5p(s, eo, x, y);
            else {
                fwdcore5p(n, s, x, y, W);
                fwdfftp(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                fwdend3p(s, eo, x, y);
            else {
                fwdcore3p(n, s, x, y, W);
                fwdfftp(n/3, 3*s, !eo, y, x, W);
            }
        }
        else if (n%7 == 0) {
            if (n == 7)
                fwdend7p(s, eo, x, y);
            else {
                fwdcore7p(n, s, x, y, W);
                fwdfftp(n/7, 7*s, !eo, y, x, W);
            }
        }
        else fwdfftanyp(7, n, s, eo, x, y, W);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Inverse Butterfly Operation with OpenMP
    ///////////////////////////////////////////////////////////////////////////////

    void invend2p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(zq+0, addpz2(a, b));
                setpz2(zq+s, subpz2(a, b));
            }
        }
        else {
#pragma omp single
            {
                const xmm a = getpz(x[0]);
                const xmm b = getpz(x[1]);
                setpz(z[0], addpz(a, b));
                setpz(z[1], subpz(a, b));
            }
        }
    }

    void invcore2p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/2;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/2;
        if (s >= 2) {
#pragma omp for schedule(static)
            for (int i = 0; i < N/4; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s2p = 2*sp;
                const ymm wp = duppz3(W[N-sp]);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s2p = y + q + s2p;
                const ymm a = getpz2(xq_sp+N0);
                const ymm b = getpz2(xq_sp+N1);
                setpz2(yq_s2p+s*0,            addpz2(a, b));
                setpz2(yq_s2p+s*1, mulpz2(wp, subpz2(a, b)));
            }
        }
        else {
#pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_2p = y + 2*p;
                const xmm wp = getpz(W[N-p]);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                setpz(y_2p[0],           addpz(a, b));
                setpz(y_2p[1], mulpz(wp, subpz(a, b)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend4p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(zq+s*0, addpz2(apc,  bpd));
                setpz2(zq+s*1, addpz2(amc, jbmd));
                setpz2(zq+s*2, subpz2(apc,  bpd));
                setpz2(zq+s*3, subpz2(amc, jbmd));
            }
        }
        else {
#pragma omp single
            {
                const xmm a = getpz(x[0]);
                const xmm b = getpz(x[1]);
                const xmm c = getpz(x[2]);
                const xmm d = getpz(x[3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(z[0], addpz(apc,  bpd));
                setpz(z[1], addpz(amc, jbmd));
                setpz(z[2], subpz(apc,  bpd));
                setpz(z[3], subpz(amc, jbmd));
            }
        }
    }

    void invcore4p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/4;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/4;
        const int N2 = N1*2;
        const int N3 = N1*3;
        if (s >= 2) {
#pragma omp for schedule(static)
            for (int i = 0; i < N/8; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s4p = 4*sp;
                const ymm w1p = duppz3(W[N-1*sp]);
                const ymm w2p = duppz3(W[N-2*sp]);
                const ymm w3p = duppz3(W[N-3*sp]);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s4p = y + q + s4p;
                const ymm a = getpz2(xq_sp+N0);
                const ymm b = getpz2(xq_sp+N1);
                const ymm c = getpz2(xq_sp+N2);
                const ymm d = getpz2(xq_sp+N3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                setpz2(yq_s4p+s*1, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(yq_s4p+s*3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
        }
        else {
#pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                const xmm w1p = cnjpz(getpz(W[p]));
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm a = getpz(x_p[N0]);
                const xmm b = getpz(x_p[N1]);
                const xmm c = getpz(x_p[N2]);
                const xmm d = getpz(x_p[N3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(y_4p[0],            addpz(apc,  bpd));
                setpz(y_4p[1], mulpz(w1p, addpz(amc, jbmd)));
                setpz(y_4p[2], mulpz(w2p, subpz(apc,  bpd)));
                setpz(y_4p[3], mulpz(w3p, subpz(amc, jbmd)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend8p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
        if (s >= 2) {
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm x0 = getpz2(xq+s*0);
                const ymm x1 = getpz2(xq+s*1);
                const ymm x2 = getpz2(xq+s*2);
                const ymm x3 = getpz2(xq+s*3);
                const ymm x4 = getpz2(xq+s*4);
                const ymm x5 = getpz2(xq+s*5);
                const ymm x6 = getpz2(xq+s*6);
                const ymm x7 = getpz2(xq+s*7);
                const ymm  a04 =       addpz2(x0, x4);
                const ymm  s04 =       subpz2(x0, x4);
                const ymm  a26 =       addpz2(x2, x6);
                const ymm js26 = jxpz2(subpz2(x2, x6));
                const ymm  a15 =       addpz2(x1, x5);
                const ymm  s15 =       subpz2(x1, x5);
                const ymm  a37 =       addpz2(x3, x7);
                const ymm js37 = jxpz2(subpz2(x3, x7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(zq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(zq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(zq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(zq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(zq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(zq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
            }
        }
        else {
#pragma omp single
            {
                const xmm x0 = getpz(x[0]);
                const xmm x1 = getpz(x[1]);
                const xmm x2 = getpz(x[2]);
                const xmm x3 = getpz(x[3]);
                const xmm x4 = getpz(x[4]);
                const xmm x5 = getpz(x[5]);
                const xmm x6 = getpz(x[6]);
                const xmm x7 = getpz(x[7]);
                const xmm  a04 =      addpz(x0, x4);
                const xmm  s04 =      subpz(x0, x4);
                const xmm  a26 =      addpz(x2, x6);
                const xmm js26 = jxpz(subpz(x2, x6));
                const xmm  a15 =      addpz(x1, x5);
                const xmm  s15 =      subpz(x1, x5);
                const xmm  a37 =      addpz(x3, x7);
                const xmm js37 = jxpz(subpz(x3, x7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                setpz(z[0], addpz(a04_p1_a26,    a15_p1_a37));
                setpz(z[1], addpz(s04_pj_s26, v8_s15_pj_s37));
                setpz(z[2], addpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(z[3], subpz(s04_mj_s26, w8_s15_mj_s37));
                setpz(z[4], subpz(a04_p1_a26,    a15_p1_a37));
                setpz(z[5], subpz(s04_pj_s26, v8_s15_pj_s37));
                setpz(z[6], subpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(z[7], addpz(s04_mj_s26, w8_s15_mj_s37));
            }
        }
    }

    void invcore8p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int m  = n/8;
        const int N  = n*s;
        const int N0 = 0;
        const int N1 = N/8;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const int N5 = N1*5;
        const int N6 = N1*6;
        const int N7 = N1*7;
        if (s >= 2) {
#pragma omp for schedule(static)
            for (int i = 0; i < N/16; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s8p = 8*sp;
                const ymm w1p = duppz3(W[N-1*sp]);
                const ymm w2p = duppz3(W[N-2*sp]);
                const ymm w3p = duppz3(W[N-3*sp]);
                const ymm w4p = mulpz2(w2p,w2p);
                const ymm w5p = mulpz2(w2p,w3p);
                const ymm w6p = mulpz2(w3p,w3p);
                const ymm w7p = mulpz2(w3p,w4p);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s8p = y + q + s8p;
                const ymm x0 = getpz2(xq_sp+N0);
                const ymm x1 = getpz2(xq_sp+N1);
                const ymm x2 = getpz2(xq_sp+N2);
                const ymm x3 = getpz2(xq_sp+N3);
                const ymm x4 = getpz2(xq_sp+N4);
                const ymm x5 = getpz2(xq_sp+N5);
                const ymm x6 = getpz2(xq_sp+N6);
                const ymm x7 = getpz2(xq_sp+N7);
                const ymm  a04 =       addpz2(x0, x4);
                const ymm  s04 =       subpz2(x0, x4);
                const ymm  a26 =       addpz2(x2, x6);
                const ymm js26 = jxpz2(subpz2(x2, x6));
                const ymm  a15 =       addpz2(x1, x5);
                const ymm  s15 =       subpz2(x1, x5);
                const ymm  a37 =       addpz2(x3, x7);
                const ymm js37 = jxpz2(subpz2(x3, x7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(yq_s8p+s*0,             addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
                setpz2(yq_s8p+s*2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
                setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
                setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
                setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
                setpz2(yq_s8p+s*6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
                setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
            }
        }
        else {
#pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                const xmm w1p = cnjpz(getpz(W[p]));
                const xmm w2p = mulpz(w1p,w1p);
                const xmm w3p = mulpz(w1p,w2p);
                const xmm w4p = mulpz(w2p,w2p);
                const xmm w5p = mulpz(w2p,w3p);
                const xmm w6p = mulpz(w3p,w3p);
                const xmm w7p = mulpz(w3p,w4p);
                const xmm x0 = getpz(x_p[N0]);
                const xmm x1 = getpz(x_p[N1]);
                const xmm x2 = getpz(x_p[N2]);
                const xmm x3 = getpz(x_p[N3]);
                const xmm x4 = getpz(x_p[N4]);
                const xmm x5 = getpz(x_p[N5]);
                const xmm x6 = getpz(x_p[N6]);
                const xmm x7 = getpz(x_p[N7]);
                const xmm  a04 =      addpz(x0, x4);
                const xmm  s04 =      subpz(x0, x4);
                const xmm  a26 =      addpz(x2, x6);
                const xmm js26 = jxpz(subpz(x2, x6));
                const xmm  a15 =      addpz(x1, x5);
                const xmm  s15 =      subpz(x1, x5);
                const xmm  a37 =      addpz(x3, x7);
                const xmm js37 = jxpz(subpz(x3, x7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                setpz(y_8p[0],            addpz(a04_p1_a26,    a15_p1_a37));
                setpz(y_8p[1], mulpz(w1p, addpz(s04_pj_s26, v8_s15_pj_s37)));
                setpz(y_8p[2], mulpz(w2p, addpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[3], mulpz(w3p, subpz(s04_mj_s26, w8_s15_mj_s37)));
                setpz(y_8p[4], mulpz(w4p, subpz(a04_p1_a26,    a15_p1_a37)));
                setpz(y_8p[5], mulpz(w5p, subpz(s04_pj_s26, v8_s15_pj_s37)));
                setpz(y_8p[6], mulpz(w6p, subpz(a04_m1_a26,  j_a15_m1_a37)));
                setpz(y_8p[7], mulpz(w7p, addpz(s04_mj_s26, w8_s15_mj_s37)));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend5p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            xmm r[5];
            bfly5<1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]), getpz(x[q+s*4]), r);
            for (int k = 0; k < 5; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void invcore5p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassYp<5, 1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/5;
        const int ms = N/5;
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[N-1*sp]);
            const cpx w2p = getpz(W[N-2*sp]);
            const cpx w3p = getpz(W[N-3*sp]);
            const cpx w4p = getpz(W[N-4*sp]);
            const int q_sp = q + sp;
            xmm r[5];
            bfly5<1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]), getpz(x[q_sp+N1*4]), r);
            const int q_s5p = q + sp*5;
            setpz(y[q_s5p+s*0], r[0]);
            setpz(y[q_s5p+s*1], cpx(r[1])*w1p);
            setpz(y[q_s5p+s*2], cpx(r[2])*w2p);
            setpz(y[q_s5p+s*3], cpx(r[3])*w3p);
            setpz(y[q_s5p+s*4], cpx(r[4])*w4p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend3p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            xmm r[3];
            bfly3<1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), r);
            for (int k = 0; k < 3; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void invcore3p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassYp<3, 1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int N1 = N/3;
        const int ms = N/3;
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[N-1*sp]);
            const cpx w2p = getpz(W[N-2*sp]);
            const int q_sp = q + sp;
            xmm r[3];
            bfly3<1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), r);
            const int q_s3p = q + sp*3;
            setpz(y[q_s3p+s*0], r[0]);
            setpz(y[q_s3p+s*1], cpx(r[1])*w1p);
            setpz(y[q_s3p+s*2], cpx(r[2])*w2p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    void invend7p(const int s, const bool eo,
                  complex_vector x, complex_vector y) noexcept
    {
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            xmm r[7];
            bfly7<1>(getpz(x[q+s*0]), getpz(x[q+s*1]), getpz(x[q+s*2]), getpz(x[q+s*3]),
                      getpz(x[q+s*4]), getpz(x[q+s*5]), getpz(x[q+s*6]), r);
            for (int k = 0; k < 7; k++) setpz(z[q+s*k], r[k]);
        }
    }

    void invcore7p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
#ifdef OTFFT_MIXED_YMM
        if ((s & 1) == 0) { corePassYp<7, 1>(n, s, x, y, W); return; }
#endif
        const int N  = n*s;
        const int ms = N/7;
        const int N1 = N/7;
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[N-1*sp]);
            const cpx w2p = getpz(W[N-2*sp]);
            const cpx w3p = getpz(W[N-3*sp]);
            const cpx w4p = getpz(W[N-4*sp]);
            const cpx w5p = getpz(W[N-5*sp]);
            const cpx w6p = getpz(W[N-6*sp]);
            const int q_sp = q + sp;
            xmm r[7];
            bfly7<1>(getpz(x[q_sp+N1*0]), getpz(x[q_sp+N1*1]), getpz(x[q_sp+N1*2]), getpz(x[q_sp+N1*3]),
                      getpz(x[q_sp+N1*4]), getpz(x[q_sp+N1*5]), getpz(x[q_sp+N1*6]), r);
            const int q_s7p = q + sp*7;
            setpz(y[q_s7p+s*0], r[0]);
            setpz(y[q_s7p+s*1], cpx(r[1])*w1p);
            setpz(y[q_s7p+s*2], cpx(r[2])*w2p);
            setpz(y[q_s7p+s*3], cpx(r[3])*w3p);
            setpz(y[q_s7p+s*4], cpx(r[4])*w4p);
            setpz(y[q_s7p+s*5], cpx(r[5])*w5p);
            setpz(y[q_s7p+s*6], cpx(r[6])*w6p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size IFFT except Radix-2,3,5,7 with OpenMP
    ///////////////////////////////////////////////////////////////////////////////

    void invfftanyp(const int r, const int n, const int s, const bool eo,
                    complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        constexpr xmm zero = { 0, 0 };
        const int N = n*s;
        int k = r;
        while (n%k != 0) {
            if (k*k > n) { k = n; break; }
            k += 2;
        }
        if (k == n) {
#pragma omp for schedule(static)
            for (int q = 0; q < s; q++) {
                for (int i = 0; i < k; i++) {
                    cpx z = zero;
                    int ij = 0;
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[N-s*ij]);
                        z = z + a*wij;
                        ij += i; if (ij >= k) ij -= k;
                    }
                    setpz(y[q+s*i], z);
                }
            }
            if (!eo) {
#pragma omp for schedule(static) nowait
                for (int p = 0; p < N; p++) setpz(x[p], getpz(y[p]));
            }
        }
        else {
            const int m  = n/k;
            const int ms = m*s;
#pragma omp for schedule(static)
            for (int h = 0; h < ms; h++) {
                const int p = h / s;
                const int q = h % s;
                const int sp = s*p;
                const int q_sp  = q + sp;
                const int q_spk = q + sp*k;
                for (int i = 0; i < k; i++) {
                    cpx z = zero;
                    int ij = 0;
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q_sp+ms*j]);
                        const cpx wij = getpz(W[N-ms*ij]);
                        z = z + a*wij;
                        ij += i; if (ij >= k) ij -= k;
                    }
                    const cpx wip = getpz(W[N-i*sp]);
                    setpz(y[q_spk+s*i], z * wip);
                }
            }
            invfftanyp(k, m, k*s, !eo, y, x, W);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Mixed Radix IFFT with OpenMP
    ///////////////////////////////////////////////////////////////////////////////

    void invfftp(const int n, const int s, const bool eo,
                 complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N = n*s;
        if (N < 2) return;
        if (n%8 == 0) {
            if (n == 8)
                invend8p(s, eo, x, y);
            else {
                invcore8p(n, s, x, y, W);
                invfftp(n/8, 8*s, !eo, y, x, W);
            }
        }
        else if (n%4 == 0) {
            if (n == 4)
                invend4p(s, eo, x, y);
            else {
                invcore4p(n, s, x, y, W);
                invfftp(n/4, 4*s, !eo, y, x, W);
            }
        }
        else if (n%2 == 0) {
            if (n == 2)
                invend2p(s, eo, x, y);
            else {
                invcore2p(n, s, x, y, W);
                invfftp(n/2, 2*s, !eo, y, x, W);
            }
        }
        else if (n%5 == 0) {
            if (n == 5)
                invend5p(s, eo, x, y);
            else {
                invcore5p(n, s, x, y, W);
                invfftp(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                invend3p(s, eo, x, y);
            else {
                invcore3p(n, s, x, y, W);
                invfftp(n/3, 3*s, !eo, y, x, W);
            }
        }
        else if (n%7 == 0) {
            if (n == 7)
                invend7p(s, eo, x, y);
            else {
                invcore7p(n, s, x, y, W);
                invfftp(n/7, 7*s, !eo, y, x, W);
            }
        }
        else invfftanyp(7, n, s, eo, x, y, W);
    }

} /////////////////////////////////////////////////////////////////////////////

}

#endif // otfft_mixedradix_h
