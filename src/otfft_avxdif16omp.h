/******************************************************************************
*  OTFFT AVXDIF(Radix-16) of OpenMP Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdif16omp_h
#define otfft_avxdif16omp_h

namespace OTFFT_NAMESPACE {

namespace OTFFT_AVXDIF16omp { /////////////////////////////////////////////////

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
        static constexpr int Ni = N1/4;
        static constexpr int h  = s/4;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
#pragma omp for schedule(static)
            for (int i = 0; i < Ni; i++) {
                const int p = i / h;
                const int q = i % h * 4;
                const int sp   = s*p;
                const int s16p = 16*sp;
                complex_vector xq_sp   = x + q + sp;
                complex_vector yq_s16p = y + q + s16p;

                const emm w1p = dupez5(W[sp]);

                const emm x0 = getez4(xq_sp+N0);
                const emm x1 = getez4(xq_sp+N1);
                const emm x2 = getez4(xq_sp+N2);
                const emm x3 = getez4(xq_sp+N3);
                const emm x4 = getez4(xq_sp+N4);
                const emm x5 = getez4(xq_sp+N5);
                const emm x6 = getez4(xq_sp+N6);
                const emm x7 = getez4(xq_sp+N7);
                const emm x8 = getez4(xq_sp+N8);
                const emm x9 = getez4(xq_sp+N9);
                const emm xa = getez4(xq_sp+Na);
                const emm xb = getez4(xq_sp+Nb);
                const emm xc = getez4(xq_sp+Nc);
                const emm xd = getez4(xq_sp+Nd);
                const emm xe = getez4(xq_sp+Ne);
                const emm xf = getez4(xq_sp+Nf);

                const emm a08 = addez4(x0, x8); const emm s08 = subez4(x0, x8);
                const emm a4c = addez4(x4, xc); const emm s4c = subez4(x4, xc);
                const emm a2a = addez4(x2, xa); const emm s2a = subez4(x2, xa);
                const emm a6e = addez4(x6, xe); const emm s6e = subez4(x6, xe);
                const emm a19 = addez4(x1, x9); const emm s19 = subez4(x1, x9);
                const emm a5d = addez4(x5, xd); const emm s5d = subez4(x5, xd);
                const emm a3b = addez4(x3, xb); const emm s3b = subez4(x3, xb);
                const emm a7f = addez4(x7, xf); const emm s7f = subez4(x7, xf);

                const emm js4c = jxez4(s4c);
                const emm js6e = jxez4(s6e);
                const emm js5d = jxez4(s5d);
                const emm js7f = jxez4(s7f);

                const emm a08p1a4c = addez4(a08, a4c); const emm s08mjs4c = subez4(s08, js4c);
                const emm a08m1a4c = subez4(a08, a4c); const emm s08pjs4c = addez4(s08, js4c);
                const emm a2ap1a6e = addez4(a2a, a6e); const emm s2amjs6e = subez4(s2a, js6e);
                const emm a2am1a6e = subez4(a2a, a6e); const emm s2apjs6e = addez4(s2a, js6e);
                const emm a19p1a5d = addez4(a19, a5d); const emm s19mjs5d = subez4(s19, js5d);
                const emm a19m1a5d = subez4(a19, a5d); const emm s19pjs5d = addez4(s19, js5d);
                const emm a3bp1a7f = addez4(a3b, a7f); const emm s3bmjs7f = subez4(s3b, js7f);
                const emm a3bm1a7f = subez4(a3b, a7f); const emm s3bpjs7f = addez4(s3b, js7f);

                const emm w8_s2amjs6e = w8xez4(s2amjs6e);
                const emm  j_a2am1a6e =  jxez4(a2am1a6e);
                const emm v8_s2apjs6e = v8xez4(s2apjs6e);

                const emm a08p1a4c_p1_a2ap1a6e = addez4(a08p1a4c,    a2ap1a6e);
                const emm s08mjs4c_pw_s2amjs6e = addez4(s08mjs4c, w8_s2amjs6e);
                const emm a08m1a4c_mj_a2am1a6e = subez4(a08m1a4c,  j_a2am1a6e);
                const emm s08pjs4c_mv_s2apjs6e = subez4(s08pjs4c, v8_s2apjs6e);
                const emm a08p1a4c_m1_a2ap1a6e = subez4(a08p1a4c,    a2ap1a6e);
                const emm s08mjs4c_mw_s2amjs6e = subez4(s08mjs4c, w8_s2amjs6e);
                const emm a08m1a4c_pj_a2am1a6e = addez4(a08m1a4c,  j_a2am1a6e);
                const emm s08pjs4c_pv_s2apjs6e = addez4(s08pjs4c, v8_s2apjs6e);

                const emm w8_s3bmjs7f = w8xez4(s3bmjs7f);
                const emm  j_a3bm1a7f =  jxez4(a3bm1a7f);
                const emm v8_s3bpjs7f = v8xez4(s3bpjs7f);

                const emm a19p1a5d_p1_a3bp1a7f = addez4(a19p1a5d,    a3bp1a7f);
                const emm s19mjs5d_pw_s3bmjs7f = addez4(s19mjs5d, w8_s3bmjs7f);
                const emm a19m1a5d_mj_a3bm1a7f = subez4(a19m1a5d,  j_a3bm1a7f);
                const emm s19pjs5d_mv_s3bpjs7f = subez4(s19pjs5d, v8_s3bpjs7f);
                const emm a19p1a5d_m1_a3bp1a7f = subez4(a19p1a5d,    a3bp1a7f);
                const emm s19mjs5d_mw_s3bmjs7f = subez4(s19mjs5d, w8_s3bmjs7f);
                const emm a19m1a5d_pj_a3bm1a7f = addez4(a19m1a5d,  j_a3bm1a7f);
                const emm s19pjs5d_pv_s3bpjs7f = addez4(s19pjs5d, v8_s3bpjs7f);

                const emm h1_s19mjs5d_pw_s3bmjs7f = h1xez4(s19mjs5d_pw_s3bmjs7f);
                const emm w8_a19m1a5d_mj_a3bm1a7f = w8xez4(a19m1a5d_mj_a3bm1a7f);
                const emm h3_s19pjs5d_mv_s3bpjs7f = h3xez4(s19pjs5d_mv_s3bpjs7f);
                const emm  j_a19p1a5d_m1_a3bp1a7f =  jxez4(a19p1a5d_m1_a3bp1a7f);
                const emm hd_s19mjs5d_mw_s3bmjs7f = hdxez4(s19mjs5d_mw_s3bmjs7f);
                const emm v8_a19m1a5d_pj_a3bm1a7f = v8xez4(a19m1a5d_pj_a3bm1a7f);
                const emm hf_s19pjs5d_pv_s3bpjs7f = hfxez4(s19pjs5d_pv_s3bpjs7f);

                const emm w2p = mulez4(w1p,w1p);
                const emm w3p = mulez4(w1p,w2p);
                const emm w4p = mulez4(w2p,w2p);
                const emm w5p = mulez4(w2p,w3p);
                const emm w6p = mulez4(w3p,w3p);
                const emm w7p = mulez4(w3p,w4p);
                const emm w8p = mulez4(w4p,w4p);
                const emm w9p = mulez4(w4p,w5p);
                const emm wap = mulez4(w5p,w5p);
                const emm wbp = mulez4(w5p,w6p);
                const emm wcp = mulez4(w6p,w6p);
                const emm wdp = mulez4(w6p,w7p);
                const emm wep = mulez4(w7p,w7p);
                const emm wfp = mulez4(w7p,w8p);

                setez4(yq_s16p+s*0x0,             addez4(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setez4(yq_s16p+s*0x1, mulez4(w1p, addez4(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
                setez4(yq_s16p+s*0x2, mulez4(w2p, addez4(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setez4(yq_s16p+s*0x3, mulez4(w3p, addez4(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setez4(yq_s16p+s*0x4, mulez4(w4p, subez4(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setez4(yq_s16p+s*0x5, mulez4(w5p, subez4(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setez4(yq_s16p+s*0x6, mulez4(w6p, subez4(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setez4(yq_s16p+s*0x7, mulez4(w7p, subez4(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));

                setez4(yq_s16p+s*0x8, mulez4(w8p, subez4(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
                setez4(yq_s16p+s*0x9, mulez4(w9p, subez4(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
                setez4(yq_s16p+s*0xa, mulez4(wap, subez4(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setez4(yq_s16p+s*0xb, mulez4(wbp, subez4(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setez4(yq_s16p+s*0xc, mulez4(wcp, addez4(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setez4(yq_s16p+s*0xd, mulez4(wdp, addez4(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setez4(yq_s16p+s*0xe, mulez4(wep, addez4(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setez4(yq_s16p+s*0xf, mulez4(wfp, addez4(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
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
#pragma omp for schedule(static)
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p   = x + p;
                complex_vector y_16p = y + 16*p;

                const ymm w1p = getpz2(W+p);

                const ymm x0 = getpz2(x_p+N0);
                const ymm x1 = getpz2(x_p+N1);
                const ymm x2 = getpz2(x_p+N2);
                const ymm x3 = getpz2(x_p+N3);
                const ymm x4 = getpz2(x_p+N4);
                const ymm x5 = getpz2(x_p+N5);
                const ymm x6 = getpz2(x_p+N6);
                const ymm x7 = getpz2(x_p+N7);
                const ymm x8 = getpz2(x_p+N8);
                const ymm x9 = getpz2(x_p+N9);
                const ymm xa = getpz2(x_p+Na);
                const ymm xb = getpz2(x_p+Nb);
                const ymm xc = getpz2(x_p+Nc);
                const ymm xd = getpz2(x_p+Nd);
                const ymm xe = getpz2(x_p+Ne);
                const ymm xf = getpz2(x_p+Nf);

                const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
                const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
                const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
                const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
                const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
                const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
                const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
                const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

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

                const ymm w2p = mulpz2(w1p,w1p);
                const ymm w3p = mulpz2(w1p,w2p);
                const ymm w4p = mulpz2(w2p,w2p);
                const ymm w5p = mulpz2(w2p,w3p);
                const ymm w6p = mulpz2(w3p,w3p);
                const ymm w7p = mulpz2(w3p,w4p);
                const ymm w8p = mulpz2(w4p,w4p);
                const ymm w9p = mulpz2(w4p,w5p);
                const ymm wap = mulpz2(w5p,w5p);
                const ymm wbp = mulpz2(w5p,w6p);
                const ymm wcp = mulpz2(w6p,w6p);
                const ymm wdp = mulpz2(w6p,w7p);
                const ymm wep = mulpz2(w7p,w7p);
                const ymm wfp = mulpz2(w7p,w8p);
                const ymm aA =             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f);
                const ymm bB = mulpz2(w1p, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                const ymm cC = mulpz2(w2p, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                const ymm dD = mulpz2(w3p, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                const ymm eE = mulpz2(w4p, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                const ymm fF = mulpz2(w5p, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                const ymm gG = mulpz2(w6p, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                const ymm hH = mulpz2(w7p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                const ymm iI = mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                const ymm jJ = mulpz2(w9p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                const ymm kK = mulpz2(wap, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                const ymm lL = mulpz2(wbp, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                const ymm mM = mulpz2(wcp, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                const ymm nN = mulpz2(wdp, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                const ymm oO = mulpz2(wep, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                const ymm pP = mulpz2(wfp, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                const ymm ab = catlo(aA, bB);
                setpz2(y_16p+0x00, ab);
                const ymm cd = catlo(cC, dD);
                setpz2(y_16p+0x02, cd);
                const ymm ef = catlo(eE, fF);
                setpz2(y_16p+0x04, ef);
                const ymm gh = catlo(gG, hH);
                setpz2(y_16p+0x06, gh);
                const ymm ij = catlo(iI, jJ);
                setpz2(y_16p+0x08, ij);
                const ymm kl = catlo(kK, lL);
                setpz2(y_16p+0x0a, kl);
                const ymm mn = catlo(mM, nN);
                setpz2(y_16p+0x0c, mn);
                const ymm op = catlo(oO, pP);
                setpz2(y_16p+0x0e, op);
                const ymm AB = cathi(aA, bB);
                setpz2(y_16p+0x10, AB);
                const ymm CD = cathi(cC, dD);
                setpz2(y_16p+0x12, CD);
                const ymm EF = cathi(eE, fF);
                setpz2(y_16p+0x14, EF);
                const ymm GH = cathi(gG, hH);
                setpz2(y_16p+0x16, GH);
                const ymm IJ = cathi(iI, jJ);
                setpz2(y_16p+0x18, IJ);
                const ymm KL = cathi(kK, lL);
                setpz2(y_16p+0x1a, KL);
                const ymm MN = cathi(mM, nN);
                setpz2(y_16p+0x1c, MN);
                const ymm OP = cathi(oO, pP);
                setpz2(y_16p+0x1e, OP);
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
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;

                const ymm x0 = scalepz2<N,mode>(getpz2(xq+s*0x0));
                const ymm x1 = scalepz2<N,mode>(getpz2(xq+s*0x1));
                const ymm x2 = scalepz2<N,mode>(getpz2(xq+s*0x2));
                const ymm x3 = scalepz2<N,mode>(getpz2(xq+s*0x3));
                const ymm x4 = scalepz2<N,mode>(getpz2(xq+s*0x4));
                const ymm x5 = scalepz2<N,mode>(getpz2(xq+s*0x5));
                const ymm x6 = scalepz2<N,mode>(getpz2(xq+s*0x6));
                const ymm x7 = scalepz2<N,mode>(getpz2(xq+s*0x7));
                const ymm x8 = scalepz2<N,mode>(getpz2(xq+s*0x8));
                const ymm x9 = scalepz2<N,mode>(getpz2(xq+s*0x9));
                const ymm xa = scalepz2<N,mode>(getpz2(xq+s*0xa));
                const ymm xb = scalepz2<N,mode>(getpz2(xq+s*0xb));
                const ymm xc = scalepz2<N,mode>(getpz2(xq+s*0xc));
                const ymm xd = scalepz2<N,mode>(getpz2(xq+s*0xd));
                const ymm xe = scalepz2<N,mode>(getpz2(xq+s*0xe));
                const ymm xf = scalepz2<N,mode>(getpz2(xq+s*0xf));

                const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
                const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
                const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
                const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
                const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
                const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
                const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
                const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

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

                setpz2(zq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(zq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(zq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(zq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(zq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(zq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(zq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(zq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                setpz2(zq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(zq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz2(zq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(zq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(zq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(zq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(zq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(zq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
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
                const xmm x0 = scalepz<16,mode>(getpz(x[0x0]));
                const xmm x1 = scalepz<16,mode>(getpz(x[0x1]));
                const xmm x2 = scalepz<16,mode>(getpz(x[0x2]));
                const xmm x3 = scalepz<16,mode>(getpz(x[0x3]));
                const xmm x4 = scalepz<16,mode>(getpz(x[0x4]));
                const xmm x5 = scalepz<16,mode>(getpz(x[0x5]));
                const xmm x6 = scalepz<16,mode>(getpz(x[0x6]));
                const xmm x7 = scalepz<16,mode>(getpz(x[0x7]));
                const xmm x8 = scalepz<16,mode>(getpz(x[0x8]));
                const xmm x9 = scalepz<16,mode>(getpz(x[0x9]));
                const xmm xa = scalepz<16,mode>(getpz(x[0xa]));
                const xmm xb = scalepz<16,mode>(getpz(x[0xb]));
                const xmm xc = scalepz<16,mode>(getpz(x[0xc]));
                const xmm xd = scalepz<16,mode>(getpz(x[0xd]));
                const xmm xe = scalepz<16,mode>(getpz(x[0xe]));
                const xmm xf = scalepz<16,mode>(getpz(x[0xf]));

                const xmm a08 = addpz(x0, x8); const xmm s08 = subpz(x0, x8);
                const xmm a4c = addpz(x4, xc); const xmm s4c = subpz(x4, xc);
                const xmm a2a = addpz(x2, xa); const xmm s2a = subpz(x2, xa);
                const xmm a6e = addpz(x6, xe); const xmm s6e = subpz(x6, xe);
                const xmm a19 = addpz(x1, x9); const xmm s19 = subpz(x1, x9);
                const xmm a5d = addpz(x5, xd); const xmm s5d = subpz(x5, xd);
                const xmm a3b = addpz(x3, xb); const xmm s3b = subpz(x3, xb);
                const xmm a7f = addpz(x7, xf); const xmm s7f = subpz(x7, xf);

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

                setpz(z[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(z[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz(z[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(z[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(z[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(z[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(z[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(z[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

                setpz(z[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(z[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
                setpz(z[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(z[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(z[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(z[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(z[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(z[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
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
            fwdcore<n,s>()(x, y, W);
            fwdfft<n/16,16*s,!eo,mode>()(y, x, W);
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
            OTFFT_AVXDIF8omp::fwdend<8,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<4,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIF4omp::fwdend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIF4omp::fwdend<2,s,eo,mode>()(x, y);
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
        static constexpr int Ni = N1/4;
        static constexpr int h  = s/4;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
#pragma omp for schedule(static)
            for (int i = 0; i < Ni; i++) {
                const int p = i / h;
                const int q = i % h * 4;
                const int sp   = s*p;
                const int s16p = 16*sp;
                complex_vector xq_sp   = x + q + sp;
                complex_vector yq_s16p = y + q + s16p;

                const emm w1p = dupez5(conj(W[sp]));

                const emm x0 = getez4(xq_sp+N0);
                const emm x1 = getez4(xq_sp+N1);
                const emm x2 = getez4(xq_sp+N2);
                const emm x3 = getez4(xq_sp+N3);
                const emm x4 = getez4(xq_sp+N4);
                const emm x5 = getez4(xq_sp+N5);
                const emm x6 = getez4(xq_sp+N6);
                const emm x7 = getez4(xq_sp+N7);
                const emm x8 = getez4(xq_sp+N8);
                const emm x9 = getez4(xq_sp+N9);
                const emm xa = getez4(xq_sp+Na);
                const emm xb = getez4(xq_sp+Nb);
                const emm xc = getez4(xq_sp+Nc);
                const emm xd = getez4(xq_sp+Nd);
                const emm xe = getez4(xq_sp+Ne);
                const emm xf = getez4(xq_sp+Nf);

                const emm a08 = addez4(x0, x8); const emm s08 = subez4(x0, x8);
                const emm a4c = addez4(x4, xc); const emm s4c = subez4(x4, xc);
                const emm a2a = addez4(x2, xa); const emm s2a = subez4(x2, xa);
                const emm a6e = addez4(x6, xe); const emm s6e = subez4(x6, xe);
                const emm a19 = addez4(x1, x9); const emm s19 = subez4(x1, x9);
                const emm a5d = addez4(x5, xd); const emm s5d = subez4(x5, xd);
                const emm a3b = addez4(x3, xb); const emm s3b = subez4(x3, xb);
                const emm a7f = addez4(x7, xf); const emm s7f = subez4(x7, xf);

                const emm js4c = jxez4(s4c);
                const emm js6e = jxez4(s6e);
                const emm js5d = jxez4(s5d);
                const emm js7f = jxez4(s7f);

                const emm a08p1a4c = addez4(a08, a4c); const emm s08mjs4c = subez4(s08, js4c);
                const emm a08m1a4c = subez4(a08, a4c); const emm s08pjs4c = addez4(s08, js4c);
                const emm a2ap1a6e = addez4(a2a, a6e); const emm s2amjs6e = subez4(s2a, js6e);
                const emm a2am1a6e = subez4(a2a, a6e); const emm s2apjs6e = addez4(s2a, js6e);
                const emm a19p1a5d = addez4(a19, a5d); const emm s19mjs5d = subez4(s19, js5d);
                const emm a19m1a5d = subez4(a19, a5d); const emm s19pjs5d = addez4(s19, js5d);
                const emm a3bp1a7f = addez4(a3b, a7f); const emm s3bmjs7f = subez4(s3b, js7f);
                const emm a3bm1a7f = subez4(a3b, a7f); const emm s3bpjs7f = addez4(s3b, js7f);

                const emm w8_s2amjs6e = w8xez4(s2amjs6e);
                const emm  j_a2am1a6e =  jxez4(a2am1a6e);
                const emm v8_s2apjs6e = v8xez4(s2apjs6e);

                const emm a08p1a4c_p1_a2ap1a6e = addez4(a08p1a4c,    a2ap1a6e);
                const emm s08mjs4c_pw_s2amjs6e = addez4(s08mjs4c, w8_s2amjs6e);
                const emm a08m1a4c_mj_a2am1a6e = subez4(a08m1a4c,  j_a2am1a6e);
                const emm s08pjs4c_mv_s2apjs6e = subez4(s08pjs4c, v8_s2apjs6e);
                const emm a08p1a4c_m1_a2ap1a6e = subez4(a08p1a4c,    a2ap1a6e);
                const emm s08mjs4c_mw_s2amjs6e = subez4(s08mjs4c, w8_s2amjs6e);
                const emm a08m1a4c_pj_a2am1a6e = addez4(a08m1a4c,  j_a2am1a6e);
                const emm s08pjs4c_pv_s2apjs6e = addez4(s08pjs4c, v8_s2apjs6e);

                const emm w8_s3bmjs7f = w8xez4(s3bmjs7f);
                const emm  j_a3bm1a7f =  jxez4(a3bm1a7f);
                const emm v8_s3bpjs7f = v8xez4(s3bpjs7f);

                const emm a19p1a5d_p1_a3bp1a7f = addez4(a19p1a5d,    a3bp1a7f);
                const emm s19mjs5d_pw_s3bmjs7f = addez4(s19mjs5d, w8_s3bmjs7f);
                const emm a19m1a5d_mj_a3bm1a7f = subez4(a19m1a5d,  j_a3bm1a7f);
                const emm s19pjs5d_mv_s3bpjs7f = subez4(s19pjs5d, v8_s3bpjs7f);
                const emm a19p1a5d_m1_a3bp1a7f = subez4(a19p1a5d,    a3bp1a7f);
                const emm s19mjs5d_mw_s3bmjs7f = subez4(s19mjs5d, w8_s3bmjs7f);
                const emm a19m1a5d_pj_a3bm1a7f = addez4(a19m1a5d,  j_a3bm1a7f);
                const emm s19pjs5d_pv_s3bpjs7f = addez4(s19pjs5d, v8_s3bpjs7f);

                const emm h1_s19mjs5d_pw_s3bmjs7f = h1xez4(s19mjs5d_pw_s3bmjs7f);
                const emm w8_a19m1a5d_mj_a3bm1a7f = w8xez4(a19m1a5d_mj_a3bm1a7f);
                const emm h3_s19pjs5d_mv_s3bpjs7f = h3xez4(s19pjs5d_mv_s3bpjs7f);
                const emm  j_a19p1a5d_m1_a3bp1a7f =  jxez4(a19p1a5d_m1_a3bp1a7f);
                const emm hd_s19mjs5d_mw_s3bmjs7f = hdxez4(s19mjs5d_mw_s3bmjs7f);
                const emm v8_a19m1a5d_pj_a3bm1a7f = v8xez4(a19m1a5d_pj_a3bm1a7f);
                const emm hf_s19pjs5d_pv_s3bpjs7f = hfxez4(s19pjs5d_pv_s3bpjs7f);

                const emm w2p = mulez4(w1p,w1p);
                const emm w3p = mulez4(w1p,w2p);
                const emm w4p = mulez4(w2p,w2p);
                const emm w5p = mulez4(w2p,w3p);
                const emm w6p = mulez4(w3p,w3p);
                const emm w7p = mulez4(w3p,w4p);
                const emm w8p = mulez4(w4p,w4p);
                const emm w9p = mulez4(w4p,w5p);
                const emm wap = mulez4(w5p,w5p);
                const emm wbp = mulez4(w5p,w6p);
                const emm wcp = mulez4(w6p,w6p);
                const emm wdp = mulez4(w6p,w7p);
                const emm wep = mulez4(w7p,w7p);
                const emm wfp = mulez4(w7p,w8p);

                setez4(yq_s16p+s*0x0,             addez4(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setez4(yq_s16p+s*0x1, mulez4(w1p, addez4(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
                setez4(yq_s16p+s*0x2, mulez4(w2p, addez4(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setez4(yq_s16p+s*0x3, mulez4(w3p, addez4(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setez4(yq_s16p+s*0x4, mulez4(w4p, addez4(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setez4(yq_s16p+s*0x5, mulez4(w5p, subez4(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setez4(yq_s16p+s*0x6, mulez4(w6p, subez4(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setez4(yq_s16p+s*0x7, mulez4(w7p, subez4(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));

                setez4(yq_s16p+s*0x8, mulez4(w8p, subez4(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
                setez4(yq_s16p+s*0x9, mulez4(w9p, subez4(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
                setez4(yq_s16p+s*0xa, mulez4(wap, subez4(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setez4(yq_s16p+s*0xb, mulez4(wbp, subez4(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setez4(yq_s16p+s*0xc, mulez4(wcp, subez4(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setez4(yq_s16p+s*0xd, mulez4(wdp, addez4(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setez4(yq_s16p+s*0xe, mulez4(wep, addez4(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setez4(yq_s16p+s*0xf, mulez4(wfp, addez4(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
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
#pragma omp for schedule(static)
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p   = x + p;
                complex_vector y_16p = y + 16*p;

                const ymm w1p = cnjpz2(getpz2(W+p));

                const ymm x0 = getpz2(x_p+N0);
                const ymm x1 = getpz2(x_p+N1);
                const ymm x2 = getpz2(x_p+N2);
                const ymm x3 = getpz2(x_p+N3);
                const ymm x4 = getpz2(x_p+N4);
                const ymm x5 = getpz2(x_p+N5);
                const ymm x6 = getpz2(x_p+N6);
                const ymm x7 = getpz2(x_p+N7);
                const ymm x8 = getpz2(x_p+N8);
                const ymm x9 = getpz2(x_p+N9);
                const ymm xa = getpz2(x_p+Na);
                const ymm xb = getpz2(x_p+Nb);
                const ymm xc = getpz2(x_p+Nc);
                const ymm xd = getpz2(x_p+Nd);
                const ymm xe = getpz2(x_p+Ne);
                const ymm xf = getpz2(x_p+Nf);

                const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
                const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
                const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
                const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
                const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
                const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
                const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
                const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

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

                const ymm w2p = mulpz2(w1p,w1p);
                const ymm w3p = mulpz2(w1p,w2p);
                const ymm w4p = mulpz2(w2p,w2p);
                const ymm w5p = mulpz2(w2p,w3p);
                const ymm w6p = mulpz2(w3p,w3p);
                const ymm w7p = mulpz2(w3p,w4p);
                const ymm w8p = mulpz2(w4p,w4p);
                const ymm w9p = mulpz2(w4p,w5p);
                const ymm wap = mulpz2(w5p,w5p);
                const ymm wbp = mulpz2(w5p,w6p);
                const ymm wcp = mulpz2(w6p,w6p);
                const ymm wdp = mulpz2(w6p,w7p);
                const ymm wep = mulpz2(w7p,w7p);
                const ymm wfp = mulpz2(w7p,w8p);
                const ymm aA =             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f);
                const ymm bB = mulpz2(w1p, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                const ymm cC = mulpz2(w2p, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                const ymm dD = mulpz2(w3p, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                const ymm eE = mulpz2(w4p, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                const ymm fF = mulpz2(w5p, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                const ymm gG = mulpz2(w6p, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                const ymm hH = mulpz2(w7p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                const ymm iI = mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                const ymm jJ = mulpz2(w9p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                const ymm kK = mulpz2(wap, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                const ymm lL = mulpz2(wbp, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                const ymm mM = mulpz2(wcp, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                const ymm nN = mulpz2(wdp, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                const ymm oO = mulpz2(wep, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                const ymm pP = mulpz2(wfp, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                const ymm ab = catlo(aA, bB);
                setpz2(y_16p+0x00, ab);
                const ymm cd = catlo(cC, dD);
                setpz2(y_16p+0x02, cd);
                const ymm ef = catlo(eE, fF);
                setpz2(y_16p+0x04, ef);
                const ymm gh = catlo(gG, hH);
                setpz2(y_16p+0x06, gh);
                const ymm ij = catlo(iI, jJ);
                setpz2(y_16p+0x08, ij);
                const ymm kl = catlo(kK, lL);
                setpz2(y_16p+0x0a, kl);
                const ymm mn = catlo(mM, nN);
                setpz2(y_16p+0x0c, mn);
                const ymm op = catlo(oO, pP);
                setpz2(y_16p+0x0e, op);
                const ymm AB = cathi(aA, bB);
                setpz2(y_16p+0x10, AB);
                const ymm CD = cathi(cC, dD);
                setpz2(y_16p+0x12, CD);
                const ymm EF = cathi(eE, fF);
                setpz2(y_16p+0x14, EF);
                const ymm GH = cathi(gG, hH);
                setpz2(y_16p+0x16, GH);
                const ymm IJ = cathi(iI, jJ);
                setpz2(y_16p+0x18, IJ);
                const ymm KL = cathi(kK, lL);
                setpz2(y_16p+0x1a, KL);
                const ymm MN = cathi(mM, nN);
                setpz2(y_16p+0x1c, MN);
                const ymm OP = cathi(oO, pP);
                setpz2(y_16p+0x1e, OP);
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
#pragma omp for schedule(static) nowait
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;

                const ymm x0 = scalepz2<N,mode>(getpz2(xq+s*0x0));
                const ymm x1 = scalepz2<N,mode>(getpz2(xq+s*0x1));
                const ymm x2 = scalepz2<N,mode>(getpz2(xq+s*0x2));
                const ymm x3 = scalepz2<N,mode>(getpz2(xq+s*0x3));
                const ymm x4 = scalepz2<N,mode>(getpz2(xq+s*0x4));
                const ymm x5 = scalepz2<N,mode>(getpz2(xq+s*0x5));
                const ymm x6 = scalepz2<N,mode>(getpz2(xq+s*0x6));
                const ymm x7 = scalepz2<N,mode>(getpz2(xq+s*0x7));
                const ymm x8 = scalepz2<N,mode>(getpz2(xq+s*0x8));
                const ymm x9 = scalepz2<N,mode>(getpz2(xq+s*0x9));
                const ymm xa = scalepz2<N,mode>(getpz2(xq+s*0xa));
                const ymm xb = scalepz2<N,mode>(getpz2(xq+s*0xb));
                const ymm xc = scalepz2<N,mode>(getpz2(xq+s*0xc));
                const ymm xd = scalepz2<N,mode>(getpz2(xq+s*0xd));
                const ymm xe = scalepz2<N,mode>(getpz2(xq+s*0xe));
                const ymm xf = scalepz2<N,mode>(getpz2(xq+s*0xf));

                const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
                const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
                const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
                const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
                const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
                const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
                const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
                const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

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

                setpz2(zq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(zq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(zq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(zq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(zq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(zq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(zq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(zq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                setpz2(zq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(zq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz2(zq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz2(zq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz2(zq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz2(zq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz2(zq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz2(zq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
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
                const xmm x0 = scalepz<16,mode>(getpz(x[0x0]));
                const xmm x1 = scalepz<16,mode>(getpz(x[0x1]));
                const xmm x2 = scalepz<16,mode>(getpz(x[0x2]));
                const xmm x3 = scalepz<16,mode>(getpz(x[0x3]));
                const xmm x4 = scalepz<16,mode>(getpz(x[0x4]));
                const xmm x5 = scalepz<16,mode>(getpz(x[0x5]));
                const xmm x6 = scalepz<16,mode>(getpz(x[0x6]));
                const xmm x7 = scalepz<16,mode>(getpz(x[0x7]));
                const xmm x8 = scalepz<16,mode>(getpz(x[0x8]));
                const xmm x9 = scalepz<16,mode>(getpz(x[0x9]));
                const xmm xa = scalepz<16,mode>(getpz(x[0xa]));
                const xmm xb = scalepz<16,mode>(getpz(x[0xb]));
                const xmm xc = scalepz<16,mode>(getpz(x[0xc]));
                const xmm xd = scalepz<16,mode>(getpz(x[0xd]));
                const xmm xe = scalepz<16,mode>(getpz(x[0xe]));
                const xmm xf = scalepz<16,mode>(getpz(x[0xf]));

                const xmm a08 = addpz(x0, x8); const xmm s08 = subpz(x0, x8);
                const xmm a4c = addpz(x4, xc); const xmm s4c = subpz(x4, xc);
                const xmm a2a = addpz(x2, xa); const xmm s2a = subpz(x2, xa);
                const xmm a6e = addpz(x6, xe); const xmm s6e = subpz(x6, xe);
                const xmm a19 = addpz(x1, x9); const xmm s19 = subpz(x1, x9);
                const xmm a5d = addpz(x5, xd); const xmm s5d = subpz(x5, xd);
                const xmm a3b = addpz(x3, xb); const xmm s3b = subpz(x3, xb);
                const xmm a7f = addpz(x7, xf); const xmm s7f = subpz(x7, xf);

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

                setpz(z[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(z[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz(z[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(z[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(z[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(z[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(z[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(z[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

                setpz(z[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz(z[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
                setpz(z[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
                setpz(z[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
                setpz(z[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
                setpz(z[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
                setpz(z[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
                setpz(z[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
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
            invcore<n,s>()(x, y, W);
            invfft<n/16,16*s,!eo,mode>()(y, x, W);
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
            OTFFT_AVXDIF8omp::invend<8,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct invfft<4,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIF4omp::invend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct invfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIF4omp::invend<2,s,eo,mode>()(x, y);
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

#endif // otfft_avxdif16omp_h
