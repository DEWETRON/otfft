// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

#include "otfft_misc.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_AVXDIT8omp { //////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;

    ///////////////////////////////////////////////////////////////////////////////
    // Forward Buffterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s> struct fwdcore
    {
        static const int N  = n*s;
        static const int N0 = 0;
        static const int N1 = N/8;
        static const int N2 = N1*2;
        static const int N3 = N1*3;
        static const int N4 = N1*4;
        static const int N5 = N1*5;
        static const int N6 = N1*6;
        static const int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
#pragma omp for schedule(static)
            for (int i = 0; i < N/16; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s8p = 8*sp;
                //const ymm w1p = duppz2(getpz(W[sp]));
                const ymm w1p = duppz3(W[1*sp]);
                const ymm w2p = duppz3(W[2*sp]);
                const ymm w3p = duppz3(W[3*sp]);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s8p = y + q + s8p;
                const ymm y0 =             getpz2(yq_s8p+s*0);
                const ymm y1 = mulpz2(w1p, getpz2(yq_s8p+s*1));
                const ymm y2 = mulpz2(w2p, getpz2(yq_s8p+s*2));
                const ymm y3 = mulpz2(w3p, getpz2(yq_s8p+s*3));
                const ymm y4 = mulpz2(w4p, getpz2(yq_s8p+s*4));
                const ymm y5 = mulpz2(w5p, getpz2(yq_s8p+s*5));
                const ymm y6 = mulpz2(w6p, getpz2(yq_s8p+s*6));
                const ymm y7 = mulpz2(w7p, getpz2(yq_s8p+s*7));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(xq_sp+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N1, addpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq_sp+N2, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N3, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq_sp+N4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N5, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq_sp+N6, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N7, addpz2(s04_pj_s26, v8_s15_pj_s37));
            }
        }
    };

    template <int N> struct fwdcore<N,1>
    {
        static const int N0 = 0;
        static const int N1 = N/8;
        static const int N2 = N1*2;
        static const int N3 = N1*3;
        static const int N4 = N1*4;
        static const int N5 = N1*5;
        static const int N6 = N1*6;
        static const int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
#pragma omp for schedule(static) nowait
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                const ymm w1p = getpz2(W+p);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm ab = getpz2(y_8p+ 0);
                const ymm cd = getpz2(y_8p+ 2);
                const ymm ef = getpz2(y_8p+ 4);
                const ymm gh = getpz2(y_8p+ 6);
                const ymm AB = getpz2(y_8p+ 8);
                const ymm CD = getpz2(y_8p+10);
                const ymm EF = getpz2(y_8p+12);
                const ymm GH = getpz2(y_8p+14);
                const ymm y0 =             catlo(ab, AB);
                const ymm y1 = mulpz2(w1p, cathi(ab, AB));
                const ymm y2 = mulpz2(w2p, catlo(cd, CD));
                const ymm y3 = mulpz2(w3p, cathi(cd, CD));
                const ymm y4 = mulpz2(w4p, catlo(ef, EF));
                const ymm y5 = mulpz2(w5p, cathi(ef, EF));
                const ymm y6 = mulpz2(w6p, catlo(gh, GH));
                const ymm y7 = mulpz2(w7p, cathi(gh, GH));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(x_p+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(x_p+N1, addpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(x_p+N2, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(x_p+N3, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(x_p+N4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(x_p+N5, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(x_p+N6, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(x_p+N7, addpz2(s04_pj_s26, v8_s15_pj_s37));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct fwdend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct fwdend<8,s,eo,mode>
    {
        static const int N = 8*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
#pragma omp for schedule(static)
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm y0 = scalepz2<N,mode>(getpz2(zq+s*0));
                const ymm y1 = scalepz2<N,mode>(getpz2(zq+s*1));
                const ymm y2 = scalepz2<N,mode>(getpz2(zq+s*2));
                const ymm y3 = scalepz2<N,mode>(getpz2(zq+s*3));
                const ymm y4 = scalepz2<N,mode>(getpz2(zq+s*4));
                const ymm y5 = scalepz2<N,mode>(getpz2(zq+s*5));
                const ymm y6 = scalepz2<N,mode>(getpz2(zq+s*6));
                const ymm y7 = scalepz2<N,mode>(getpz2(zq+s*7));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
            }
        }
    };

    template <bool eo, int mode> struct fwdend<8,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
#pragma omp single
            {
                zeroupper();
                complex_vector z = eo ? y : x;
                const xmm y0 = scalepz<8,mode>(getpz(z[0]));
                const xmm y1 = scalepz<8,mode>(getpz(z[1]));
                const xmm y2 = scalepz<8,mode>(getpz(z[2]));
                const xmm y3 = scalepz<8,mode>(getpz(z[3]));
                const xmm y4 = scalepz<8,mode>(getpz(z[4]));
                const xmm y5 = scalepz<8,mode>(getpz(z[5]));
                const xmm y6 = scalepz<8,mode>(getpz(z[6]));
                const xmm y7 = scalepz<8,mode>(getpz(z[7]));
                const xmm  a04 =      addpz(y0, y4);
                const xmm  s04 =      subpz(y0, y4);
                const xmm  a26 =      addpz(y2, y6);
                const xmm js26 = jxpz(subpz(y2, y6));
                const xmm  a15 =      addpz(y1, y5);
                const xmm  s15 =      subpz(y1, y5);
                const xmm  a37 =      addpz(y3, y7);
                const xmm js37 = jxpz(subpz(y3, y7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
                setpz(x[1], addpz(s04_mj_s26, w8_s15_mj_s37));
                setpz(x[2], subpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(x[3], subpz(s04_pj_s26, v8_s15_pj_s37));
                setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
                setpz(x[5], subpz(s04_mj_s26, w8_s15_mj_s37));
                setpz(x[6], addpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(x[7], addpz(s04_pj_s26, v8_s15_pj_s37));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Forward FFT
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct fwdfft
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            fwdfft<n/8,8*s,!eo,mode>()(y, x, W);
            fwdcore<n,s>()(x, y, W);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<8,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            fwdend<8,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<4,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT4omp::fwdend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT4omp::fwdend<2,s,eo,mode>()(x, y);
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Inverse Butterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s> struct invcore
    {
        static const int N  = n*s;
        static const int N0 = 0;
        static const int N1 = N/8;
        static const int N2 = N1*2;
        static const int N3 = N1*3;
        static const int N4 = N1*4;
        static const int N5 = N1*5;
        static const int N6 = N1*6;
        static const int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
#pragma omp for schedule(static)
            for (int i = 0; i < N/16; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp = s*p;
                const int s8p = 8*sp;
                //const ymm w1p = duppz2(getpz(W[N-sp]));
                const ymm w1p = duppz3(W[N-1*sp]);
                const ymm w2p = duppz3(W[N-2*sp]);
                const ymm w3p = duppz3(W[N-3*sp]);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s8p = y + q + s8p;
                const ymm y0 =             getpz2(yq_s8p+s*0);
                const ymm y1 = mulpz2(w1p, getpz2(yq_s8p+s*1));
                const ymm y2 = mulpz2(w2p, getpz2(yq_s8p+s*2));
                const ymm y3 = mulpz2(w3p, getpz2(yq_s8p+s*3));
                const ymm y4 = mulpz2(w4p, getpz2(yq_s8p+s*4));
                const ymm y5 = mulpz2(w5p, getpz2(yq_s8p+s*5));
                const ymm y6 = mulpz2(w6p, getpz2(yq_s8p+s*6));
                const ymm y7 = mulpz2(w7p, getpz2(yq_s8p+s*7));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(xq_sp+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N1, addpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq_sp+N2, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N3, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq_sp+N4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N5, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq_sp+N6, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N7, addpz2(s04_mj_s26, w8_s15_mj_s37));
            }
        }
    };

    template <int N> struct invcore<N,1>
    {
        static const int N0 = 0;
        static const int N1 = N/8;
        static const int N2 = N1*2;
        static const int N3 = N1*3;
        static const int N4 = N1*4;
        static const int N5 = N1*5;
        static const int N6 = N1*6;
        static const int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            //const_complex_vector WN = W + N;
#pragma omp for schedule(static) nowait
            for (int p = 0; p < N/8; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                //const ymm w1p = getwp2<-1>(WN,p);
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm ab = getpz2(y_8p+ 0);
                const ymm cd = getpz2(y_8p+ 2);
                const ymm ef = getpz2(y_8p+ 4);
                const ymm gh = getpz2(y_8p+ 6);
                const ymm AB = getpz2(y_8p+ 8);
                const ymm CD = getpz2(y_8p+10);
                const ymm EF = getpz2(y_8p+12);
                const ymm GH = getpz2(y_8p+14);
                const ymm y0 =             catlo(ab, AB);
                const ymm y1 = mulpz2(w1p, cathi(ab, AB));
                const ymm y2 = mulpz2(w2p, catlo(cd, CD));
                const ymm y3 = mulpz2(w3p, cathi(cd, CD));
                const ymm y4 = mulpz2(w4p, catlo(ef, EF));
                const ymm y5 = mulpz2(w5p, cathi(ef, EF));
                const ymm y6 = mulpz2(w6p, catlo(gh, GH));
                const ymm y7 = mulpz2(w7p, cathi(gh, GH));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(x_p+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(x_p+N1, addpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(x_p+N2, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(x_p+N3, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(x_p+N4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(x_p+N5, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(x_p+N6, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(x_p+N7, addpz2(s04_mj_s26, w8_s15_mj_s37));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct invend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct invend<8,s,eo,mode>
    {
        static const int N  = 8*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
#pragma omp for schedule(static)
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm y0 = scalepz2<N,mode>(getpz2(zq+s*0));
                const ymm y1 = scalepz2<N,mode>(getpz2(zq+s*1));
                const ymm y2 = scalepz2<N,mode>(getpz2(zq+s*2));
                const ymm y3 = scalepz2<N,mode>(getpz2(zq+s*3));
                const ymm y4 = scalepz2<N,mode>(getpz2(zq+s*4));
                const ymm y5 = scalepz2<N,mode>(getpz2(zq+s*5));
                const ymm y6 = scalepz2<N,mode>(getpz2(zq+s*6));
                const ymm y7 = scalepz2<N,mode>(getpz2(zq+s*7));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
            }
        }
    };

    template <bool eo, int mode> struct invend<8,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
#pragma omp single
            {
                zeroupper();
                complex_vector z = eo ? y : x;
                const xmm y0 = scalepz<8,mode>(getpz(z[0]));
                const xmm y1 = scalepz<8,mode>(getpz(z[1]));
                const xmm y2 = scalepz<8,mode>(getpz(z[2]));
                const xmm y3 = scalepz<8,mode>(getpz(z[3]));
                const xmm y4 = scalepz<8,mode>(getpz(z[4]));
                const xmm y5 = scalepz<8,mode>(getpz(z[5]));
                const xmm y6 = scalepz<8,mode>(getpz(z[6]));
                const xmm y7 = scalepz<8,mode>(getpz(z[7]));
                const xmm  a04 =      addpz(y0, y4);
                const xmm  s04 =      subpz(y0, y4);
                const xmm  a26 =      addpz(y2, y6);
                const xmm js26 = jxpz(subpz(y2, y6));
                const xmm  a15 =      addpz(y1, y5);
                const xmm  s15 =      subpz(y1, y5);
                const xmm  a37 =      addpz(y3, y7);
                const xmm js37 = jxpz(subpz(y3, y7));
                const xmm    a04_p1_a26 =       addpz(a04,  a26);
                const xmm    s04_pj_s26 =       addpz(s04, js26);
                const xmm    a04_m1_a26 =       subpz(a04,  a26);
                const xmm    s04_mj_s26 =       subpz(s04, js26);
                const xmm    a15_p1_a37 =       addpz(a15,  a37);
                const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
                const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
                const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
                setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
                setpz(x[1], addpz(s04_pj_s26, v8_s15_pj_s37));
                setpz(x[2], addpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(x[3], subpz(s04_mj_s26, w8_s15_mj_s37));
                setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
                setpz(x[5], subpz(s04_pj_s26, v8_s15_pj_s37));
                setpz(x[6], subpz(a04_m1_a26,  j_a15_m1_a37));
                setpz(x[7], addpz(s04_mj_s26, w8_s15_mj_s37));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Inverse FFT
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct invfft
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            invfft<n/8,8*s,!eo,mode>()(y, x, W);
            invcore<n,s>()(x, y, W);
        }
    };

    template <int s, bool eo, int mode> struct invfft<8,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            invend<8,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct invfft<4,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT4omp::invend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct invfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT4omp::invend<2,s,eo,mode>()(x, y);
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Power of 2 FFT Routine
    ///////////////////////////////////////////////////////////////////////////////

    inline void fwd(const int log_N,
                    complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        static const int mode = scale_length;
#pragma omp parallel firstprivate(x,y,W)
        switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y, W); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y, W); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y, W); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y, W); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y, W); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y, W); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y, W); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y, W); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y, W); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y, W); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y, W); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y, W); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y, W); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y, W); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y, W); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y, W); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y, W); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y, W); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y, W); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y, W); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y, W); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y, W); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y, W); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y, W); break;
        }
    }

    inline void fwd0(const int log_N,
                     complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        static const int mode = scale_1;
#pragma omp parallel firstprivate(x,y,W)
        switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y, W); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y, W); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y, W); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y, W); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y, W); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y, W); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y, W); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y, W); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y, W); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y, W); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y, W); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y, W); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y, W); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y, W); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y, W); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y, W); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y, W); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y, W); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y, W); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y, W); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y, W); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y, W); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y, W); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y, W); break;
        }
    }

    inline void fwdu(const int log_N,
                     complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        static const int mode = scale_unitary;
#pragma omp parallel firstprivate(x,y,W)
        switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y, W); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y, W); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y, W); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y, W); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y, W); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y, W); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y, W); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y, W); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y, W); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y, W); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y, W); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y, W); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y, W); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y, W); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y, W); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y, W); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y, W); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y, W); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y, W); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y, W); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y, W); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y, W); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y, W); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y, W); break;
        }
    }

    inline void fwdn(const int log_N,
                     complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        fwd(log_N, x, y, W);
    }

    ///////////////////////////////////////////////////////////////////////////////

    inline void inv(const int log_N,
                    complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        static const int mode = scale_1;
#pragma omp parallel firstprivate(x,y,W)
        switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y, W); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y, W); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y, W); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y, W); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y, W); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y, W); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y, W); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y, W); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y, W); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y, W); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y, W); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y, W); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y, W); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y, W); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y, W); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y, W); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y, W); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y, W); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y, W); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y, W); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y, W); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y, W); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y, W); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y, W); break;
        }
    }

    inline void inv0(const int log_N,
                     complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        inv(log_N, x, y, W);
    }

    inline void invu(const int log_N,
                     complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        static const int mode = scale_unitary;
#pragma omp parallel firstprivate(x,y,W)
        switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y, W); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y, W); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y, W); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y, W); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y, W); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y, W); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y, W); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y, W); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y, W); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y, W); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y, W); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y, W); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y, W); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y, W); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y, W); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y, W); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y, W); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y, W); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y, W); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y, W); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y, W); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y, W); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y, W); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y, W); break;
        }
    }

    inline void invn(const int log_N,
                     complex_vector x, complex_vector y, const_complex_vector W) noexcept
    {
        static const int mode = scale_length;
#pragma omp parallel firstprivate(x,y,W)
        switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y, W); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y, W); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y, W); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y, W); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y, W); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y, W); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y, W); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y, W); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y, W); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y, W); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y, W); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y, W); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y, W); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y, W); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y, W); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y, W); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y, W); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y, W); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y, W); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y, W); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y, W); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y, W); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y, W); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y, W); break;
        }
    }

} /////////////////////////////////////////////////////////////////////////////

}
