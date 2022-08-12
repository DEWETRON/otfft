// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

#include "otfft_misc.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_AVXDIT16omp { /////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;

    ///////////////////////////////////////////////////////////////////////////////
    // Forward Buffterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s> struct fwdcore
    {
        static constexpr int N  = n*s;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/16;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;
        static constexpr int N8 = N1*8;
        static constexpr int N9 = N1*9;
        static constexpr int Na = N1*10;
        static constexpr int Nb = N1*11;
        static constexpr int Nc = N1*12;
        static constexpr int Nd = N1*13;
        static constexpr int Ne = N1*14;
        static constexpr int Nf = N1*15;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < N/32; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp   = s*p;
                const int s16p = 16*sp;

                complex_vector xq_sp   = x + q + sp;
                complex_vector yq_s16p = y + q + s16p;

                const ymm y0 =             getpz2(yq_s16p+s*0x0);
                const ymm w1p = duppz3(W[sp]);
                const ymm y1 = mulpz2(w1p, getpz2(yq_s16p+s*0x1));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm y2 = mulpz2(w2p, getpz2(yq_s16p+s*0x2));
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm y3 = mulpz2(w3p, getpz2(yq_s16p+s*0x3));
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm y4 = mulpz2(w4p, getpz2(yq_s16p+s*0x4));
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm y5 = mulpz2(w5p, getpz2(yq_s16p+s*0x5));
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm y6 = mulpz2(w6p, getpz2(yq_s16p+s*0x6));
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm y7 = mulpz2(w7p, getpz2(yq_s16p+s*0x7));
                const ymm w8p = mulpz2(w4p, w4p);
                const ymm y8 = mulpz2(w8p, getpz2(yq_s16p+s*0x8));
                const ymm w9p = mulpz2(w4p, w5p);
                const ymm y9 = mulpz2(w9p, getpz2(yq_s16p+s*0x9));
                const ymm wap = mulpz2(w5p, w5p);
                const ymm ya = mulpz2(wap, getpz2(yq_s16p+s*0xa));
                const ymm wbp = mulpz2(w5p, w6p);
                const ymm yb = mulpz2(wbp, getpz2(yq_s16p+s*0xb));
                const ymm wcp = mulpz2(w6p, w6p);
                const ymm yc = mulpz2(wcp, getpz2(yq_s16p+s*0xc));
                const ymm wdp = mulpz2(w6p, w7p);
                const ymm yd = mulpz2(wdp, getpz2(yq_s16p+s*0xd));
                const ymm wep = mulpz2(w7p, w7p);
                const ymm ye = mulpz2(wep, getpz2(yq_s16p+s*0xe));
                const ymm wfp = mulpz2(w7p, w8p);
                const ymm yf = mulpz2(wfp, getpz2(yq_s16p+s*0xf));

                const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
                const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
                const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
                const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
                const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
                const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
                const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
                const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

                const ymm js4c = jxpz2(s4c);
                const ymm js6e = jxpz2(s6e);
                const ymm js5d = jxpz2(s5d);
                const ymm js7f = jxpz2(s7f);

                const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
                const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
                const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
                const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
                const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
                const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
                const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
                const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

                const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
                const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
                const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

                const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
                const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

                const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
                const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
                const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

                const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
                const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

                const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
                const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
                const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
                const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
                const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
                const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
                const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

                setpz2(xq_sp+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq_sp+N1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(xq_sp+N2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq_sp+N3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq_sp+N4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq_sp+N5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq_sp+N6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq_sp+N7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                setpz2(xq_sp+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq_sp+N9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(xq_sp+Na, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq_sp+Nb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq_sp+Nc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq_sp+Nd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq_sp+Ne, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq_sp+Nf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            }
        }
    };

    template <int N> struct fwdcore<N,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = N/16;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;
        static constexpr int N8 = N1*8;
        static constexpr int N9 = N1*9;
        static constexpr int Na = N1*10;
        static constexpr int Nb = N1*11;
        static constexpr int Nc = N1*12;
        static constexpr int Nd = N1*13;
        static constexpr int Ne = N1*14;
        static constexpr int Nf = N1*15;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            #pragma omp for schedule(static) nowait
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p   = x + p;
                complex_vector y_16p = y + 16*p;

                const ymm w1p = getpz2(W+p);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm w8p = mulpz2(w4p, w4p);
                const ymm w9p = mulpz2(w4p, w5p);
                const ymm wap = mulpz2(w5p, w5p);
                const ymm wbp = mulpz2(w5p, w6p);
                const ymm wcp = mulpz2(w6p, w6p);
                const ymm wdp = mulpz2(w6p, w7p);
                const ymm wep = mulpz2(w7p, w7p);
                const ymm wfp = mulpz2(w7p, w8p);
                const ymm ab = getpz2(y_16p+0x00);
                const ymm cd = getpz2(y_16p+0x02);
                const ymm ef = getpz2(y_16p+0x04);
                const ymm gh = getpz2(y_16p+0x06);
                const ymm ij = getpz2(y_16p+0x08);
                const ymm kl = getpz2(y_16p+0x0a);
                const ymm mn = getpz2(y_16p+0x0c);
                const ymm op = getpz2(y_16p+0x0e);

                const ymm AB = getpz2(y_16p+0x10);
                const ymm CD = getpz2(y_16p+0x12);
                const ymm EF = getpz2(y_16p+0x14);
                const ymm GH = getpz2(y_16p+0x16);
                const ymm IJ = getpz2(y_16p+0x18);
                const ymm KL = getpz2(y_16p+0x1a);
                const ymm MN = getpz2(y_16p+0x1c);
                const ymm OP = getpz2(y_16p+0x1e);

                const ymm y0 =             catlo(ab, AB);
                const ymm y1 = mulpz2(w1p, cathi(ab, AB));
                const ymm y2 = mulpz2(w2p, catlo(cd, CD));
                const ymm y3 = mulpz2(w3p, cathi(cd, CD));
                const ymm y4 = mulpz2(w4p, catlo(ef, EF));
                const ymm y5 = mulpz2(w5p, cathi(ef, EF));
                const ymm y6 = mulpz2(w6p, catlo(gh, GH));
                const ymm y7 = mulpz2(w7p, cathi(gh, GH));

                const ymm y8 = mulpz2(w8p, catlo(ij, IJ));
                const ymm y9 = mulpz2(w9p, cathi(ij, IJ));
                const ymm ya = mulpz2(wap, catlo(kl, KL));
                const ymm yb = mulpz2(wbp, cathi(kl, KL));
                const ymm yc = mulpz2(wcp, catlo(mn, MN));
                const ymm yd = mulpz2(wdp, cathi(mn, MN));
                const ymm ye = mulpz2(wep, catlo(op, OP));
                const ymm yf = mulpz2(wfp, cathi(op, OP));
                const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
                const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
                const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
                const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
                const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
                const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
                const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
                const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

                const ymm js4c = jxpz2(s4c);
                const ymm js6e = jxpz2(s6e);
                const ymm js5d = jxpz2(s5d);
                const ymm js7f = jxpz2(s7f);

                const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
                const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
                const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
                const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
                const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
                const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
                const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
                const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

                const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
                const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
                const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

                const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
                const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

                const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
                const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
                const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

                const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
                const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

                const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
                const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
                const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
                const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
                const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
                const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
                const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

                setpz2(x_p+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(x_p+N1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(x_p+N2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(x_p+N3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(x_p+N4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(x_p+N5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(x_p+N6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(x_p+N7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                setpz2(x_p+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(x_p+N9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(x_p+Na, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(x_p+Nb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(x_p+Nc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(x_p+Nd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(x_p+Ne, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(x_p+Nf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct fwdend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct fwdend<16,s,eo,mode>
    {
        static constexpr int N = 16*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            #pragma omp for schedule(static)
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;

                const ymm z0 = scalepz2<N,mode>(getpz2(zq+s*0x0));
                const ymm z1 = scalepz2<N,mode>(getpz2(zq+s*0x1));
                const ymm z2 = scalepz2<N,mode>(getpz2(zq+s*0x2));
                const ymm z3 = scalepz2<N,mode>(getpz2(zq+s*0x3));
                const ymm z4 = scalepz2<N,mode>(getpz2(zq+s*0x4));
                const ymm z5 = scalepz2<N,mode>(getpz2(zq+s*0x5));
                const ymm z6 = scalepz2<N,mode>(getpz2(zq+s*0x6));
                const ymm z7 = scalepz2<N,mode>(getpz2(zq+s*0x7));
                const ymm z8 = scalepz2<N,mode>(getpz2(zq+s*0x8));
                const ymm z9 = scalepz2<N,mode>(getpz2(zq+s*0x9));
                const ymm za = scalepz2<N,mode>(getpz2(zq+s*0xa));
                const ymm zb = scalepz2<N,mode>(getpz2(zq+s*0xb));
                const ymm zc = scalepz2<N,mode>(getpz2(zq+s*0xc));
                const ymm zd = scalepz2<N,mode>(getpz2(zq+s*0xd));
                const ymm ze = scalepz2<N,mode>(getpz2(zq+s*0xe));
                const ymm zf = scalepz2<N,mode>(getpz2(zq+s*0xf));

                const ymm a08 = addpz2(z0, z8); const ymm s08 = subpz2(z0, z8);
                const ymm a4c = addpz2(z4, zc); const ymm s4c = subpz2(z4, zc);
                const ymm a2a = addpz2(z2, za); const ymm s2a = subpz2(z2, za);
                const ymm a6e = addpz2(z6, ze); const ymm s6e = subpz2(z6, ze);
                const ymm a19 = addpz2(z1, z9); const ymm s19 = subpz2(z1, z9);
                const ymm a5d = addpz2(z5, zd); const ymm s5d = subpz2(z5, zd);
                const ymm a3b = addpz2(z3, zb); const ymm s3b = subpz2(z3, zb);
                const ymm a7f = addpz2(z7, zf); const ymm s7f = subpz2(z7, zf);

                const ymm js4c = jxpz2(s4c);
                const ymm js6e = jxpz2(s6e);
                const ymm js5d = jxpz2(s5d);
                const ymm js7f = jxpz2(s7f);

                const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
                const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
                const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
                const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
                const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
                const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
                const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
                const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

                const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
                const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
                const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

                const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
                const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

                const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
                const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
                const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

                const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
                const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

                const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
                const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
                const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
                const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
                const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
                const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
                const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

                setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(xq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(xq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            }
        }
    };

    template <bool eo, int mode> struct fwdend<16,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            #pragma omp single
            {
                zeroupper();
                complex_vector z = eo ? y : x;
                const xmm z0 = scalepz<16,mode>(getpz(z[0x0]));
                const xmm z1 = scalepz<16,mode>(getpz(z[0x1]));
                const xmm z2 = scalepz<16,mode>(getpz(z[0x2]));
                const xmm z3 = scalepz<16,mode>(getpz(z[0x3]));
                const xmm z4 = scalepz<16,mode>(getpz(z[0x4]));
                const xmm z5 = scalepz<16,mode>(getpz(z[0x5]));
                const xmm z6 = scalepz<16,mode>(getpz(z[0x6]));
                const xmm z7 = scalepz<16,mode>(getpz(z[0x7]));
                const xmm z8 = scalepz<16,mode>(getpz(z[0x8]));
                const xmm z9 = scalepz<16,mode>(getpz(z[0x9]));
                const xmm za = scalepz<16,mode>(getpz(z[0xa]));
                const xmm zb = scalepz<16,mode>(getpz(z[0xb]));
                const xmm zc = scalepz<16,mode>(getpz(z[0xc]));
                const xmm zd = scalepz<16,mode>(getpz(z[0xd]));
                const xmm ze = scalepz<16,mode>(getpz(z[0xe]));
                const xmm zf = scalepz<16,mode>(getpz(z[0xf]));

                const xmm a08 = addpz(z0, z8); const xmm s08 = subpz(z0, z8);
                const xmm a4c = addpz(z4, zc); const xmm s4c = subpz(z4, zc);
                const xmm a2a = addpz(z2, za); const xmm s2a = subpz(z2, za);
                const xmm a6e = addpz(z6, ze); const xmm s6e = subpz(z6, ze);
                const xmm a19 = addpz(z1, z9); const xmm s19 = subpz(z1, z9);
                const xmm a5d = addpz(z5, zd); const xmm s5d = subpz(z5, zd);
                const xmm a3b = addpz(z3, zb); const xmm s3b = subpz(z3, zb);
                const xmm a7f = addpz(z7, zf); const xmm s7f = subpz(z7, zf);

                const xmm js4c = jxpz(s4c);
                const xmm js6e = jxpz(s6e);
                const xmm js5d = jxpz(s5d);
                const xmm js7f = jxpz(s7f);

                const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
                const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
                const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
                const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
                const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
                const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
                const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
                const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

                const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
                const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
                const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

                const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
                const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
                const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
                const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
                const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
                const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
                const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
                const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

                const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
                const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
                const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

                const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
                const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
                const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
                const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
                const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
                const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
                const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
                const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

                const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
                const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
                const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
                const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
                const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
                const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
                const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

                setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(x[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz(x[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(x[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(x[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(x[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(x[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(x[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(x[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz(x[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(x[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(x[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(x[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(x[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(x[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
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
            fwdfft<n/16,16*s,!eo,mode>()(y, x, W);
            fwdcore<n,s>()(x, y, W);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<16,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            fwdend<16,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<8,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT8omp::fwdend<8,s,eo,mode>()(x, y);
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
        static constexpr int N  = n*s;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/16;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;
        static constexpr int N8 = N1*8;
        static constexpr int N9 = N1*9;
        static constexpr int Na = N1*10;
        static constexpr int Nb = N1*11;
        static constexpr int Nc = N1*12;
        static constexpr int Nd = N1*13;
        static constexpr int Ne = N1*14;
        static constexpr int Nf = N1*15;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < N/32; i++) {
                const int p = i / (s/2);
                const int q = i % (s/2) * 2;
                const int sp   = s*p;
                const int s16p = 16*sp;

                complex_vector xq_sp   = x + q + sp;
                complex_vector yq_s16p = y + q + s16p;

                const ymm y0 =             getpz2(yq_s16p+s*0x0);
                const ymm w1p = duppz3(W[N-sp]);
                const ymm y1 = mulpz2(w1p, getpz2(yq_s16p+s*0x1));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm y2 = mulpz2(w2p, getpz2(yq_s16p+s*0x2));
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm y3 = mulpz2(w3p, getpz2(yq_s16p+s*0x3));
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm y4 = mulpz2(w4p, getpz2(yq_s16p+s*0x4));
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm y5 = mulpz2(w5p, getpz2(yq_s16p+s*0x5));
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm y6 = mulpz2(w6p, getpz2(yq_s16p+s*0x6));
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm y7 = mulpz2(w7p, getpz2(yq_s16p+s*0x7));
                const ymm w8p = mulpz2(w4p, w4p);
                const ymm y8 = mulpz2(w8p, getpz2(yq_s16p+s*0x8));
                const ymm w9p = mulpz2(w4p, w5p);
                const ymm y9 = mulpz2(w9p, getpz2(yq_s16p+s*0x9));
                const ymm wap = mulpz2(w5p, w5p);
                const ymm ya = mulpz2(wap, getpz2(yq_s16p+s*0xa));
                const ymm wbp = mulpz2(w5p, w6p);
                const ymm yb = mulpz2(wbp, getpz2(yq_s16p+s*0xb));
                const ymm wcp = mulpz2(w6p, w6p);
                const ymm yc = mulpz2(wcp, getpz2(yq_s16p+s*0xc));
                const ymm wdp = mulpz2(w6p, w7p);
                const ymm yd = mulpz2(wdp, getpz2(yq_s16p+s*0xd));
                const ymm wep = mulpz2(w7p, w7p);
                const ymm ye = mulpz2(wep, getpz2(yq_s16p+s*0xe));
                const ymm wfp = mulpz2(w7p, w8p);
                const ymm yf = mulpz2(wfp, getpz2(yq_s16p+s*0xf));

                const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
                const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
                const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
                const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
                const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
                const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
                const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
                const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

                const ymm js4c = jxpz2(s4c);
                const ymm js6e = jxpz2(s6e);
                const ymm js5d = jxpz2(s5d);
                const ymm js7f = jxpz2(s7f);

                const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
                const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
                const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
                const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
                const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
                const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
                const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
                const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

                const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
                const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
                const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

                const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
                const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

                const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
                const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
                const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

                const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
                const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

                const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
                const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
                const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
                const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
                const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
                const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
                const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

                setpz2(xq_sp+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq_sp+N1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(xq_sp+N2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq_sp+N3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq_sp+N4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq_sp+N5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq_sp+N6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq_sp+N7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                setpz2(xq_sp+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq_sp+N9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(xq_sp+Na, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq_sp+Nb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq_sp+Nc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq_sp+Nd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq_sp+Ne, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq_sp+Nf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            }
        }
    };

    template <int N> struct invcore<N,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = N/16;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;
        static constexpr int N8 = N1*8;
        static constexpr int N9 = N1*9;
        static constexpr int Na = N1*10;
        static constexpr int Nb = N1*11;
        static constexpr int Nc = N1*12;
        static constexpr int Nd = N1*13;
        static constexpr int Ne = N1*14;
        static constexpr int Nf = N1*15;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            #pragma omp for schedule(static) nowait
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p   = x + p;
                complex_vector y_16p = y + 16*p;

                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm w8p = mulpz2(w4p, w4p);
                const ymm w9p = mulpz2(w4p, w5p);
                const ymm wap = mulpz2(w5p, w5p);
                const ymm wbp = mulpz2(w5p, w6p);
                const ymm wcp = mulpz2(w6p, w6p);
                const ymm wdp = mulpz2(w6p, w7p);
                const ymm wep = mulpz2(w7p, w7p);
                const ymm wfp = mulpz2(w7p, w8p);
                const ymm ab = getpz2(y_16p+0x00);
                const ymm cd = getpz2(y_16p+0x02);
                const ymm ef = getpz2(y_16p+0x04);
                const ymm gh = getpz2(y_16p+0x06);
                const ymm ij = getpz2(y_16p+0x08);
                const ymm kl = getpz2(y_16p+0x0a);
                const ymm mn = getpz2(y_16p+0x0c);
                const ymm op = getpz2(y_16p+0x0e);

                const ymm AB = getpz2(y_16p+0x10);
                const ymm CD = getpz2(y_16p+0x12);
                const ymm EF = getpz2(y_16p+0x14);
                const ymm GH = getpz2(y_16p+0x16);
                const ymm IJ = getpz2(y_16p+0x18);
                const ymm KL = getpz2(y_16p+0x1a);
                const ymm MN = getpz2(y_16p+0x1c);
                const ymm OP = getpz2(y_16p+0x1e);

                const ymm y0 =             catlo(ab, AB);
                const ymm y1 = mulpz2(w1p, cathi(ab, AB));
                const ymm y2 = mulpz2(w2p, catlo(cd, CD));
                const ymm y3 = mulpz2(w3p, cathi(cd, CD));
                const ymm y4 = mulpz2(w4p, catlo(ef, EF));
                const ymm y5 = mulpz2(w5p, cathi(ef, EF));
                const ymm y6 = mulpz2(w6p, catlo(gh, GH));
                const ymm y7 = mulpz2(w7p, cathi(gh, GH));

                const ymm y8 = mulpz2(w8p, catlo(ij, IJ));
                const ymm y9 = mulpz2(w9p, cathi(ij, IJ));
                const ymm ya = mulpz2(wap, catlo(kl, KL));
                const ymm yb = mulpz2(wbp, cathi(kl, KL));
                const ymm yc = mulpz2(wcp, catlo(mn, MN));
                const ymm yd = mulpz2(wdp, cathi(mn, MN));
                const ymm ye = mulpz2(wep, catlo(op, OP));
                const ymm yf = mulpz2(wfp, cathi(op, OP));
                const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
                const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
                const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
                const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
                const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
                const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
                const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
                const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

                const ymm js4c = jxpz2(s4c);
                const ymm js6e = jxpz2(s6e);
                const ymm js5d = jxpz2(s5d);
                const ymm js7f = jxpz2(s7f);

                const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
                const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
                const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
                const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
                const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
                const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
                const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
                const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

                const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
                const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
                const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

                const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
                const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

                const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
                const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
                const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

                const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
                const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

                const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
                const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
                const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
                const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
                const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
                const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
                const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

                setpz2(x_p+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(x_p+N1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(x_p+N2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(x_p+N3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(x_p+N4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(x_p+N5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(x_p+N6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(x_p+N7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                setpz2(x_p+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(x_p+N9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(x_p+Na, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(x_p+Nb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(x_p+Nc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(x_p+Nd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(x_p+Ne, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(x_p+Nf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct invend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct invend<16,s,eo,mode>
    {
        static constexpr int N = 16*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            #pragma omp for schedule(static)
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;

                const ymm z0 = scalepz2<N,mode>(getpz2(zq+s*0x0));
                const ymm z1 = scalepz2<N,mode>(getpz2(zq+s*0x1));
                const ymm z2 = scalepz2<N,mode>(getpz2(zq+s*0x2));
                const ymm z3 = scalepz2<N,mode>(getpz2(zq+s*0x3));
                const ymm z4 = scalepz2<N,mode>(getpz2(zq+s*0x4));
                const ymm z5 = scalepz2<N,mode>(getpz2(zq+s*0x5));
                const ymm z6 = scalepz2<N,mode>(getpz2(zq+s*0x6));
                const ymm z7 = scalepz2<N,mode>(getpz2(zq+s*0x7));
                const ymm z8 = scalepz2<N,mode>(getpz2(zq+s*0x8));
                const ymm z9 = scalepz2<N,mode>(getpz2(zq+s*0x9));
                const ymm za = scalepz2<N,mode>(getpz2(zq+s*0xa));
                const ymm zb = scalepz2<N,mode>(getpz2(zq+s*0xb));
                const ymm zc = scalepz2<N,mode>(getpz2(zq+s*0xc));
                const ymm zd = scalepz2<N,mode>(getpz2(zq+s*0xd));
                const ymm ze = scalepz2<N,mode>(getpz2(zq+s*0xe));
                const ymm zf = scalepz2<N,mode>(getpz2(zq+s*0xf));

                const ymm a08 = addpz2(z0, z8); const ymm s08 = subpz2(z0, z8);
                const ymm a4c = addpz2(z4, zc); const ymm s4c = subpz2(z4, zc);
                const ymm a2a = addpz2(z2, za); const ymm s2a = subpz2(z2, za);
                const ymm a6e = addpz2(z6, ze); const ymm s6e = subpz2(z6, ze);
                const ymm a19 = addpz2(z1, z9); const ymm s19 = subpz2(z1, z9);
                const ymm a5d = addpz2(z5, zd); const ymm s5d = subpz2(z5, zd);
                const ymm a3b = addpz2(z3, zb); const ymm s3b = subpz2(z3, zb);
                const ymm a7f = addpz2(z7, zf); const ymm s7f = subpz2(z7, zf);

                const ymm js4c = jxpz2(s4c);
                const ymm js6e = jxpz2(s6e);
                const ymm js5d = jxpz2(s5d);
                const ymm js7f = jxpz2(s7f);

                const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
                const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
                const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
                const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
                const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
                const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
                const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
                const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

                const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
                const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
                const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

                const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
                const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
                const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
                const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
                const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

                const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
                const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
                const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

                const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
                const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
                const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
                const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
                const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

                const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
                const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
                const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
                const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
                const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
                const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
                const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

                setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(xq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(xq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(xq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(xq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(xq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(xq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(xq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(xq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            }
        }
    };

    template <bool eo, int mode> struct invend<16,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            #pragma omp single
            {
                zeroupper();
                complex_vector z = eo ? y : x;
                const xmm z0 = scalepz<16,mode>(getpz(z[0x0]));
                const xmm z1 = scalepz<16,mode>(getpz(z[0x1]));
                const xmm z2 = scalepz<16,mode>(getpz(z[0x2]));
                const xmm z3 = scalepz<16,mode>(getpz(z[0x3]));
                const xmm z4 = scalepz<16,mode>(getpz(z[0x4]));
                const xmm z5 = scalepz<16,mode>(getpz(z[0x5]));
                const xmm z6 = scalepz<16,mode>(getpz(z[0x6]));
                const xmm z7 = scalepz<16,mode>(getpz(z[0x7]));
                const xmm z8 = scalepz<16,mode>(getpz(z[0x8]));
                const xmm z9 = scalepz<16,mode>(getpz(z[0x9]));
                const xmm za = scalepz<16,mode>(getpz(z[0xa]));
                const xmm zb = scalepz<16,mode>(getpz(z[0xb]));
                const xmm zc = scalepz<16,mode>(getpz(z[0xc]));
                const xmm zd = scalepz<16,mode>(getpz(z[0xd]));
                const xmm ze = scalepz<16,mode>(getpz(z[0xe]));
                const xmm zf = scalepz<16,mode>(getpz(z[0xf]));

                const xmm a08 = addpz(z0, z8); const xmm s08 = subpz(z0, z8);
                const xmm a4c = addpz(z4, zc); const xmm s4c = subpz(z4, zc);
                const xmm a2a = addpz(z2, za); const xmm s2a = subpz(z2, za);
                const xmm a6e = addpz(z6, ze); const xmm s6e = subpz(z6, ze);
                const xmm a19 = addpz(z1, z9); const xmm s19 = subpz(z1, z9);
                const xmm a5d = addpz(z5, zd); const xmm s5d = subpz(z5, zd);
                const xmm a3b = addpz(z3, zb); const xmm s3b = subpz(z3, zb);
                const xmm a7f = addpz(z7, zf); const xmm s7f = subpz(z7, zf);

                const xmm js4c = jxpz(s4c);
                const xmm js6e = jxpz(s6e);
                const xmm js5d = jxpz(s5d);
                const xmm js7f = jxpz(s7f);

                const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
                const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
                const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
                const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
                const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
                const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
                const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
                const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

                const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
                const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
                const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

                const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
                const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
                const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
                const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
                const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
                const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
                const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
                const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

                const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
                const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
                const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

                const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
                const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
                const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
                const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
                const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
                const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
                const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
                const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

                const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
                const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
                const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
                const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
                const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
                const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
                const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

                setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(x[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz(x[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(x[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(x[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(x[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(x[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(x[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(x[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz(x[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(x[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(x[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(x[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(x[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(x[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
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
            invfft<n/16,16*s,!eo,mode>()(y, x, W);
            invcore<n,s>()(x, y, W);
        }
    };

    template <int s, bool eo, int mode> struct invfft<16,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            invend<16,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct invfft<8,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT8omp::invend<8,s,eo,mode>()(x, y);
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
        constexpr int mode = scale_length;
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
        constexpr int mode = scale_1;
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
        constexpr int mode = scale_unitary;
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
        constexpr int mode = scale_1;
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
        constexpr int mode = scale_unitary;
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
        constexpr int mode = scale_length;
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
