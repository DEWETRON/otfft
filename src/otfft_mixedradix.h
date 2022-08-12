// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

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
                 complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[s]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            const cpx d = getpz(x[q+s*3]);
            const cpx e = getpz(x[q+s*4]);
            setpz(z[q+s*0], a + b + c + d + e);
            setpz(z[q+s*1], a + w1*b + w2*c + w3*d + w4*e);
            setpz(z[q+s*2], a + w2*b + w4*c + w1*d + w3*e);
            setpz(z[q+s*3], a + w3*b + w1*c + w4*d + w2*e);
            setpz(z[q+s*4], a + w4*b + w3*c + w2*d + w1*e);
        }
    }

    void fwdcore5(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int m  = n/5;
        const int N0 = 0;
        const int N1 = N/5;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const cpx w1 = getpz(W[N1]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[sp]);
            const cpx w2p = mulpz(w1p,w1p);
            const cpx w3p = mulpz(w1p,w2p);
            const cpx w4p = mulpz(w2p,w2p);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                const cpx a = getpz(x[q_sp+N0]);
                const cpx b = getpz(x[q_sp+N1]);
                const cpx c = getpz(x[q_sp+N2]);
                const cpx d = getpz(x[q_sp+N3]);
                const cpx e = getpz(x[q_sp+N4]);
                const int q_s5p = q + sp*5;
                setpz(y[q_s5p+s*0],  a + b + c + d + e);
                setpz(y[q_s5p+s*1], (a + w1*b + w2*c + w3*d + w4*e)*w1p);
                setpz(y[q_s5p+s*2], (a + w2*b + w4*c + w1*d + w3*e)*w2p);
                setpz(y[q_s5p+s*3], (a + w3*b + w1*c + w4*d + w2*e)*w3p);
                setpz(y[q_s5p+s*4], (a + w4*b + w3*c + w2*d + w1*e)*w4p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend3(const int s, const bool eo,
                 complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[s]);
        const cpx w2 = mulpz(w1,w1);
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            setpz(z[q+s*0], a + b + c);
            setpz(z[q+s*1], a + w1*b + w2*c);
            setpz(z[q+s*2], a + w2*b + w1*c);
        }
    }

    void fwdcore3(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int m  = n/3;
        const int N0 = 0;
        const int N1 = N/3;
        const int N2 = N1*2;
        const cpx w1 = getpz(W[N1]);
        const cpx w2 = mulpz(w1,w1);
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[sp]);
            const cpx w2p = mulpz(w1p,w1p);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                const cpx a = getpz(x[q_sp+N0]);
                const cpx b = getpz(x[q_sp+N1]);
                const cpx c = getpz(x[q_sp+N2]);
                const int q_s3p = q + sp*3;
                setpz(y[q_s3p+s*0],  a + b + c);
                setpz(y[q_s3p+s*1], (a + w1*b + w2*c)*w1p);
                setpz(y[q_s3p+s*2], (a + w2*b + w1*c)*w2p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size FFT except Radix-2,3,5
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
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[s*((i*j)%k)]);
                        z = z + a*wij;
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
                        for (int j = 0; j < k; j++) {
                            const cpx a   = getpz(x[q_sp+ms*j]);
                            const cpx wij = getpz(W[ms*((i*j)%k)]);
                            z = z + a*wij;
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
                fwdend5(s, eo, x, y, W);
            else {
                fwdcore5(n, s, x, y, W);
                fwdfft(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                fwdend3(s, eo, x, y, W);
            else {
                fwdcore3(n, s, x, y, W);
                fwdfft(n/3, 3*s, !eo, y, x, W);
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
                 complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[4*s]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            const cpx d = getpz(x[q+s*3]);
            const cpx e = getpz(x[q+s*4]);
            setpz(z[q+s*0], a + b + c + d + e);
            setpz(z[q+s*1], a + w1*b + w2*c + w3*d + w4*e);
            setpz(z[q+s*2], a + w2*b + w4*c + w1*d + w3*e);
            setpz(z[q+s*3], a + w3*b + w1*c + w4*d + w2*e);
            setpz(z[q+s*4], a + w4*b + w3*c + w2*d + w1*e);
        }
    }

    void invcore5(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int m  = n/5;
        const int N0 = 0;
        const int N1 = N/5;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const cpx w1 = getpz(W[N4]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[N-sp]);
            const cpx w2p = mulpz(w1p,w1p);
            const cpx w3p = mulpz(w1p,w2p);
            const cpx w4p = mulpz(w2p,w2p);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                const cpx a = getpz(x[q_sp+N0]);
                const cpx b = getpz(x[q_sp+N1]);
                const cpx c = getpz(x[q_sp+N2]);
                const cpx d = getpz(x[q_sp+N3]);
                const cpx e = getpz(x[q_sp+N4]);
                const int q_s5p = q + sp*5;
                setpz(y[q_s5p+s*0],  a + b + c + d + e);
                setpz(y[q_s5p+s*1], (a + w1*b + w2*c + w3*d + w4*e)*w1p);
                setpz(y[q_s5p+s*2], (a + w2*b + w4*c + w1*d + w3*e)*w2p);
                setpz(y[q_s5p+s*3], (a + w3*b + w1*c + w4*d + w2*e)*w3p);
                setpz(y[q_s5p+s*4], (a + w4*b + w3*c + w2*d + w1*e)*w4p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend3(const int s, const bool eo,
                 complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[2*s]);
        const cpx w2 = mulpz(w1,w1);
        complex_vector z = eo ? y : x;
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            setpz(z[q+s*0], a + b + c);
            setpz(z[q+s*1], a + w1*b + w2*c);
            setpz(z[q+s*2], a + w2*b + w1*c);
        }
    }

    void invcore3(const int n, const int s,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int m  = n/3;
        const int N0 = 0;
        const int N1 = N/3;
        const int N2 = N1*2;
        const cpx w1 = getpz(W[N2]);
        const cpx w2 = mulpz(w1,w1);
        for (int p = 0; p < m; p++) {
            const int sp = s*p;
            const cpx w1p = getpz(W[N-sp]);
            const cpx w2p = mulpz(w1p,w1p);
            for (int q = 0; q < s; q++) {
                const int q_sp = q + sp;
                const cpx a = getpz(x[q_sp+N0]);
                const cpx b = getpz(x[q_sp+N1]);
                const cpx c = getpz(x[q_sp+N2]);
                const int q_s3p = q + sp*3;
                setpz(y[q_s3p+s*0],  a + b + c);
                setpz(y[q_s3p+s*1], (a + w1*b + w2*c)*w1p);
                setpz(y[q_s3p+s*2], (a + w2*b + w1*c)*w2p);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size IFFT except Radix-2,3,5
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
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[N-s*((i*j)%k)]);
                        z = z + a*wij;
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
                        for (int j = 0; j < k; j++) {
                            const cpx a   = getpz(x[q_sp+ms*j]);
                            const cpx wij = getpz(W[N-ms*((i*j)%k)]);
                            z = z + a*wij;
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
                invend5(s, eo, x, y, W);
            else {
                invcore5(n, s, x, y, W);
                invfft(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                invend3(s, eo, x, y, W);
            else {
                invcore3(n, s, x, y, W);
                invfft(n/3, 3*s, !eo, y, x, W);
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
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[s]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            const cpx d = getpz(x[q+s*3]);
            const cpx e = getpz(x[q+s*4]);
            setpz(z[q+s*0], a + b + c + d + e);
            setpz(z[q+s*1], a + w1*b + w2*c + w3*d + w4*e);
            setpz(z[q+s*2], a + w2*b + w4*c + w1*d + w3*e);
            setpz(z[q+s*3], a + w3*b + w1*c + w4*d + w2*e);
            setpz(z[q+s*4], a + w4*b + w3*c + w2*d + w1*e);
        }
    }

    void fwdcore5p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int ms = N/5;
        const int N0 = 0;
        const int N1 = N/5;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const cpx w1 = getpz(W[N1]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[sp]);
            const cpx w2p = mulpz(w1p,w1p);
            const cpx w3p = mulpz(w1p,w2p);
            const cpx w4p = mulpz(w2p,w2p);
            const int q_sp = q + sp;
            const cpx a = getpz(x[q_sp+N0]);
            const cpx b = getpz(x[q_sp+N1]);
            const cpx c = getpz(x[q_sp+N2]);
            const cpx d = getpz(x[q_sp+N3]);
            const cpx e = getpz(x[q_sp+N4]);
            const int q_s5p = q + sp*5;
            setpz(y[q_s5p+s*0],  a + b + c + d + e);
            setpz(y[q_s5p+s*1], (a + w1*b + w2*c + w3*d + w4*e)*w1p);
            setpz(y[q_s5p+s*2], (a + w2*b + w4*c + w1*d + w3*e)*w2p);
            setpz(y[q_s5p+s*3], (a + w3*b + w1*c + w4*d + w2*e)*w3p);
            setpz(y[q_s5p+s*4], (a + w4*b + w3*c + w2*d + w1*e)*w4p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void fwdend3p(const int s, const bool eo,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[s]);
        const cpx w2 = mulpz(w1,w1);
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            setpz(z[q+s*0], a + b + c);
            setpz(z[q+s*1], a + w1*b + w2*c);
            setpz(z[q+s*2], a + w2*b + w1*c);
        }
    }

    void fwdcore3p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int ms = N/3;
        const int N0 = 0;
        const int N1 = N/3;
        const int N2 = N1*2;
        const cpx w1 = getpz(W[N1]);
        const cpx w2 = mulpz(w1,w1);
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[sp]);
            const cpx w2p = mulpz(w1p,w1p);
            const int q_sp = q + sp;
            const cpx a = getpz(x[q_sp+N0]);
            const cpx b = getpz(x[q_sp+N1]);
            const cpx c = getpz(x[q_sp+N2]);
            const int q_s3p = q + sp*3;
            setpz(y[q_s3p+s*0],  a + b + c);
            setpz(y[q_s3p+s*1], (a + w1*b + w2*c)*w1p);
            setpz(y[q_s3p+s*2], (a + w2*b + w1*c)*w2p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size FFT except Radix-2,3,5 with OpenMP
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
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[s*((i*j)%k)]);
                        z = z + a*wij;
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
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q_sp+ms*j]);
                        const cpx wij = getpz(W[ms*((i*j)%k)]);
                        z = z + a*wij;
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
                fwdend5p(s, eo, x, y, W);
            else {
                fwdcore5p(n, s, x, y, W);
                fwdfftp(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                fwdend3p(s, eo, x, y, W);
            else {
                fwdcore3p(n, s, x, y, W);
                fwdfftp(n/3, 3*s, !eo, y, x, W);
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
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[4*s]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            const cpx d = getpz(x[q+s*3]);
            const cpx e = getpz(x[q+s*4]);
            setpz(z[q+s*0], a + b + c + d + e);
            setpz(z[q+s*1], a + w1*b + w2*c + w3*d + w4*e);
            setpz(z[q+s*2], a + w2*b + w4*c + w1*d + w3*e);
            setpz(z[q+s*3], a + w3*b + w1*c + w4*d + w2*e);
            setpz(z[q+s*4], a + w4*b + w3*c + w2*d + w1*e);
        }
    }

    void invcore5p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int ms = N/5;
        const int N0 = 0;
        const int N1 = N/5;
        const int N2 = N1*2;
        const int N3 = N1*3;
        const int N4 = N1*4;
        const cpx w1 = getpz(W[N4]);
        const cpx w2 = mulpz(w1,w1);
        const cpx w3 = mulpz(w1,w2);
        const cpx w4 = mulpz(w2,w2);
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[N-sp]);
            const cpx w2p = mulpz(w1p,w1p);
            const cpx w3p = mulpz(w1p,w2p);
            const cpx w4p = mulpz(w2p,w2p);
            const int q_sp = q + sp;
            const cpx a = getpz(x[q_sp+N0]);
            const cpx b = getpz(x[q_sp+N1]);
            const cpx c = getpz(x[q_sp+N2]);
            const cpx d = getpz(x[q_sp+N3]);
            const cpx e = getpz(x[q_sp+N4]);
            const int q_s5p = q + sp*5;
            setpz(y[q_s5p+s*0],  a + b + c + d + e);
            setpz(y[q_s5p+s*1], (a + w1*b + w2*c + w3*d + w4*e)*w1p);
            setpz(y[q_s5p+s*2], (a + w2*b + w4*c + w1*d + w3*e)*w2p);
            setpz(y[q_s5p+s*3], (a + w3*b + w1*c + w4*d + w2*e)*w3p);
            setpz(y[q_s5p+s*4], (a + w4*b + w3*c + w2*d + w1*e)*w4p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////

    void invend3p(const int s, const bool eo,
                  complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const cpx w1 = getpz(W[2*s]);
        const cpx w2 = mulpz(w1,w1);
        complex_vector z = eo ? y : x;
#pragma omp for schedule(static) nowait
        for (int q = 0; q < s; q++) {
            const cpx a = getpz(x[q+s*0]);
            const cpx b = getpz(x[q+s*1]);
            const cpx c = getpz(x[q+s*2]);
            setpz(z[q+s*0], a + b + c);
            setpz(z[q+s*1], a + w1*b + w2*c);
            setpz(z[q+s*2], a + w2*b + w1*c);
        }
    }

    void invcore3p(const int n, const int s,
                   complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        const int N  = n*s;
        const int ms = N/3;
        const int N0 = 0;
        const int N1 = N/3;
        const int N2 = N1*2;
        const cpx w1 = getpz(W[N2]);
        const cpx w2 = mulpz(w1,w1);
#pragma omp for schedule(static)
        for (int i = 0; i < ms; i++) {
            const int p = i / s;
            const int q = i % s;
            const int sp = s*p;
            const cpx w1p = getpz(W[N-sp]);
            const cpx w2p = mulpz(w1p,w1p);
            const int q_sp = q + sp;
            const cpx a = getpz(x[q_sp+N0]);
            const cpx b = getpz(x[q_sp+N1]);
            const cpx c = getpz(x[q_sp+N2]);
            const int q_s3p = q + sp*3;
            setpz(y[q_s3p+s*0],  a + b + c);
            setpz(y[q_s3p+s*1], (a + w1*b + w2*c)*w1p);
            setpz(y[q_s3p+s*2], (a + w2*b + w1*c)*w2p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Any Size IFFT except Radix-2,3,5 with OpenMP
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
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q+s*j]);
                        const cpx wij = getpz(W[N-s*((i*j)%k)]);
                        z = z + a*wij;
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
                    for (int j = 0; j < k; j++) {
                        const cpx a   = getpz(x[q_sp+ms*j]);
                        const cpx wij = getpz(W[N-ms*((i*j)%k)]);
                        z = z + a*wij;
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
                invend5p(s, eo, x, y, W);
            else {
                invcore5p(n, s, x, y, W);
                invfftp(n/5, 5*s, !eo, y, x, W);
            }
        }
        else if (n%3 == 0) {
            if (n == 3)
                invend3p(s, eo, x, y, W);
            else {
                invcore3p(n, s, x, y, W);
                invfftp(n/3, 3*s, !eo, y, x, W);
            }
        }
        else invfftanyp(7, n, s, eo, x, y, W);
    }

} /////////////////////////////////////////////////////////////////////////////

}
