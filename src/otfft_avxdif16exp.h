/******************************************************************************
*  OTFFT AVXDIF(Radix-16) of Exponential Version 11.6e
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

// Ported from upstream OTFFT 11.6e (R1.9, D19): the twiddle factors are computed in the
// registers (no weight table), which is faster than reading a table of N/8 or N values
// from memory for large N. The body is the upstream one, not re-indented.

#ifndef otfft_avxdif16exp_h
#define otfft_avxdif16exp_h

namespace OTFFT_NAMESPACE {

namespace OTFFT_AVXDIF16exp { /////////////////////////////////////////////////

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
    static constexpr int Ni = N1/16;
    static constexpr int h  = s/16;
    static constexpr double theta = -2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < Ni; i++) {
            const int p = i / h;
            const int q = i % h * 16;
            const int sp   = s*p;
            const int s16p = 16*sp;
            complex_vector xq_sp   = x + q + sp;
            complex_vector yq_s16p = y + q + s16p;

            const bmm w1p = duppz17(expj(theta*sp));

            const bmm x0  = getpz16(xq_sp+N0);
            const bmm x1  = getpz16(xq_sp+N1);
            const bmm x2  = getpz16(xq_sp+N2);
            const bmm x3  = getpz16(xq_sp+N3);
            const bmm x4  = getpz16(xq_sp+N4);
            const bmm x5  = getpz16(xq_sp+N5);
            const bmm x6  = getpz16(xq_sp+N6);
            const bmm x7  = getpz16(xq_sp+N7);
            const bmm x8  = getpz16(xq_sp+N8);
            const bmm x9  = getpz16(xq_sp+N9);
            const bmm xa  = getpz16(xq_sp+Na);
            const bmm xb  = getpz16(xq_sp+Nb);
            const bmm xc  = getpz16(xq_sp+Nc);
            const bmm xd  = getpz16(xq_sp+Nd);
            const bmm xe  = getpz16(xq_sp+Ne);
            const bmm xf  = getpz16(xq_sp+Nf);

            const bmm a08 = addpz16(x0, x8); const bmm s08 = subpz16(x0, x8);
            const bmm a4c = addpz16(x4, xc); const bmm s4c = subpz16(x4, xc);
            const bmm a2a = addpz16(x2, xa); const bmm s2a = subpz16(x2, xa);
            const bmm a6e = addpz16(x6, xe); const bmm s6e = subpz16(x6, xe);
            const bmm a19 = addpz16(x1, x9); const bmm s19 = subpz16(x1, x9);
            const bmm a5d = addpz16(x5, xd); const bmm s5d = subpz16(x5, xd);
            const bmm a3b = addpz16(x3, xb); const bmm s3b = subpz16(x3, xb);
            const bmm a7f = addpz16(x7, xf); const bmm s7f = subpz16(x7, xf);

            const bmm js4c = jxpz16(s4c);
            const bmm js6e = jxpz16(s6e);
            const bmm js5d = jxpz16(s5d);
            const bmm js7f = jxpz16(s7f);

            const bmm a08p1a4c = addpz16(a08, a4c); const bmm s08mjs4c = subpz16(s08, js4c);
            const bmm a08m1a4c = subpz16(a08, a4c); const bmm s08pjs4c = addpz16(s08, js4c);
            const bmm a2ap1a6e = addpz16(a2a, a6e); const bmm s2amjs6e = subpz16(s2a, js6e);
            const bmm a2am1a6e = subpz16(a2a, a6e); const bmm s2apjs6e = addpz16(s2a, js6e);
            const bmm a19p1a5d = addpz16(a19, a5d); const bmm s19mjs5d = subpz16(s19, js5d);
            const bmm a19m1a5d = subpz16(a19, a5d); const bmm s19pjs5d = addpz16(s19, js5d);
            const bmm a3bp1a7f = addpz16(a3b, a7f); const bmm s3bmjs7f = subpz16(s3b, js7f);
            const bmm a3bm1a7f = subpz16(a3b, a7f); const bmm s3bpjs7f = addpz16(s3b, js7f);

            const bmm w8_s2amjs6e = w8xpz16(s2amjs6e);
            const bmm  j_a2am1a6e =  jxpz16(a2am1a6e);
            const bmm v8_s2apjs6e = v8xpz16(s2apjs6e);

            const bmm a08p1a4c_p1_a2ap1a6e = addpz16(a08p1a4c,    a2ap1a6e);
            const bmm s08mjs4c_pw_s2amjs6e = addpz16(s08mjs4c, w8_s2amjs6e);
            const bmm a08m1a4c_mj_a2am1a6e = subpz16(a08m1a4c,  j_a2am1a6e);
            const bmm s08pjs4c_mv_s2apjs6e = subpz16(s08pjs4c, v8_s2apjs6e);
            const bmm a08p1a4c_m1_a2ap1a6e = subpz16(a08p1a4c,    a2ap1a6e);
            const bmm s08mjs4c_mw_s2amjs6e = subpz16(s08mjs4c, w8_s2amjs6e);
            const bmm a08m1a4c_pj_a2am1a6e = addpz16(a08m1a4c,  j_a2am1a6e);
            const bmm s08pjs4c_pv_s2apjs6e = addpz16(s08pjs4c, v8_s2apjs6e);

            const bmm w8_s3bmjs7f = w8xpz16(s3bmjs7f);
            const bmm  j_a3bm1a7f =  jxpz16(a3bm1a7f);
            const bmm v8_s3bpjs7f = v8xpz16(s3bpjs7f);

            const bmm a19p1a5d_p1_a3bp1a7f = addpz16(a19p1a5d,    a3bp1a7f);
            const bmm s19mjs5d_pw_s3bmjs7f = addpz16(s19mjs5d, w8_s3bmjs7f);
            const bmm a19m1a5d_mj_a3bm1a7f = subpz16(a19m1a5d,  j_a3bm1a7f);
            const bmm s19pjs5d_mv_s3bpjs7f = subpz16(s19pjs5d, v8_s3bpjs7f);
            const bmm a19p1a5d_m1_a3bp1a7f = subpz16(a19p1a5d,    a3bp1a7f);
            const bmm s19mjs5d_mw_s3bmjs7f = subpz16(s19mjs5d, w8_s3bmjs7f);
            const bmm a19m1a5d_pj_a3bm1a7f = addpz16(a19m1a5d,  j_a3bm1a7f);
            const bmm s19pjs5d_pv_s3bpjs7f = addpz16(s19pjs5d, v8_s3bpjs7f);

            const bmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz16(s19mjs5d_pw_s3bmjs7f);
            const bmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz16(a19m1a5d_mj_a3bm1a7f);
            const bmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz16(s19pjs5d_mv_s3bpjs7f);
            const bmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz16(a19p1a5d_m1_a3bp1a7f);
            const bmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz16(s19mjs5d_mw_s3bmjs7f);
            const bmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz16(a19m1a5d_pj_a3bm1a7f);
            const bmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz16(s19pjs5d_pv_s3bpjs7f);

            const bmm w2p = mulpz16(w1p,w1p);
            const bmm w3p = mulpz16(w1p,w2p);
            const bmm w4p = mulpz16(w2p,w2p);
            const bmm w5p = mulpz16(w2p,w3p);
            const bmm w6p = mulpz16(w3p,w3p);
            const bmm w7p = mulpz16(w3p,w4p);
            const bmm w8p = mulpz16(w4p,w4p);
            const bmm w9p = mulpz16(w4p,w5p);
            const bmm wap = mulpz16(w5p,w5p);
            const bmm wbp = mulpz16(w5p,w6p);
            const bmm wcp = mulpz16(w6p,w6p);
            const bmm wdp = mulpz16(w6p,w7p);
            const bmm wep = mulpz16(w7p,w7p);
            const bmm wfp = mulpz16(w7p,w8p);

            setpz16(yq_s16p+s*0x0,              addpz16(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz16(yq_s16p+s*0x1, mulpz16(w1p, addpz16(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
            setpz16(yq_s16p+s*0x2, mulpz16(w2p, addpz16(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz16(yq_s16p+s*0x3, mulpz16(w3p, addpz16(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz16(yq_s16p+s*0x4, mulpz16(w4p, subpz16(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz16(yq_s16p+s*0x5, mulpz16(w5p, subpz16(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz16(yq_s16p+s*0x6, mulpz16(w6p, subpz16(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz16(yq_s16p+s*0x7, mulpz16(w7p, subpz16(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));

            setpz16(yq_s16p+s*0x8, mulpz16(w8p, subpz16(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
            setpz16(yq_s16p+s*0x9, mulpz16(w9p, subpz16(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
            setpz16(yq_s16p+s*0xa, mulpz16(wap, subpz16(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz16(yq_s16p+s*0xb, mulpz16(wbp, subpz16(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz16(yq_s16p+s*0xc, mulpz16(wcp, addpz16(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz16(yq_s16p+s*0xd, mulpz16(wdp, addpz16(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz16(yq_s16p+s*0xe, mulpz16(wep, addpz16(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz16(yq_s16p+s*0xf, mulpz16(wfp, addpz16(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
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
    static constexpr double theta = -2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p   = x + p;
            complex_vector y_16p = y + 16*p;

            const double t1 = theta*p;
            const double t2 = t1 + theta;
            const ymm w1p = cmplx2(expj(t1), expj(t2));

            const ymm x0  = getpz2(x_p+N0);
            const ymm x1  = getpz2(x_p+N1);
            const ymm x2  = getpz2(x_p+N2);
            const ymm x3  = getpz2(x_p+N3);
            const ymm x4  = getpz2(x_p+N4);
            const ymm x5  = getpz2(x_p+N5);
            const ymm x6  = getpz2(x_p+N6);
            const ymm x7  = getpz2(x_p+N7);
            const ymm x8  = getpz2(x_p+N8);
            const ymm x9  = getpz2(x_p+N9);
            const ymm xa  = getpz2(x_p+Na);
            const ymm xb  = getpz2(x_p+Nb);
            const ymm xc  = getpz2(x_p+Nc);
            const ymm xd  = getpz2(x_p+Nd);
            const ymm xe  = getpz2(x_p+Ne);
            const ymm xf  = getpz2(x_p+Nf);

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
#if 0
            setpz3<16>(y_16p+0x0,             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz3<16>(y_16p+0x1, mulpz2(w1p, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
            setpz3<16>(y_16p+0x2, mulpz2(w2p, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0x3, mulpz2(w3p, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0x4, mulpz2(w4p, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0x5, mulpz2(w5p, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0x6, mulpz2(w6p, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0x7, mulpz2(w7p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));

            setpz3<16>(y_16p+0x8, mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
            setpz3<16>(y_16p+0x9, mulpz2(w9p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
            setpz3<16>(y_16p+0xa, mulpz2(wap, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0xb, mulpz2(wbp, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0xc, mulpz2(wcp, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0xd, mulpz2(wdp, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0xe, mulpz2(wep, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0xf, mulpz2(wfp, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
#else
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
#endif
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
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        fwdcore<n,s>()(x, y);
        fwdfft<n/16,16*s,!eo,mode>()(y, x);
    }
};

template <int s, bool eo, int mode> struct fwdfft<16,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        fwdend<16,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct fwdfft<8,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF8omp::fwdend<8,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct fwdfft<4,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF4omp::fwdend<4,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct fwdfft<2,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
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
    static constexpr int Ni = N1/16;
    static constexpr int h  = s/16;
    static constexpr double theta = 2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < Ni; i++) {
            const int p = i / h;
            const int q = i % h * 16;
            const int sp   = s*p;
            const int s16p = 16*sp;
            complex_vector xq_sp   = x + q + sp;
            complex_vector yq_s16p = y + q + s16p;

            const bmm w1p = duppz17(expj(theta*sp));

            const bmm x0  = getpz16(xq_sp+N0);
            const bmm x1  = getpz16(xq_sp+N1);
            const bmm x2  = getpz16(xq_sp+N2);
            const bmm x3  = getpz16(xq_sp+N3);
            const bmm x4  = getpz16(xq_sp+N4);
            const bmm x5  = getpz16(xq_sp+N5);
            const bmm x6  = getpz16(xq_sp+N6);
            const bmm x7  = getpz16(xq_sp+N7);
            const bmm x8  = getpz16(xq_sp+N8);
            const bmm x9  = getpz16(xq_sp+N9);
            const bmm xa  = getpz16(xq_sp+Na);
            const bmm xb  = getpz16(xq_sp+Nb);
            const bmm xc  = getpz16(xq_sp+Nc);
            const bmm xd  = getpz16(xq_sp+Nd);
            const bmm xe  = getpz16(xq_sp+Ne);
            const bmm xf  = getpz16(xq_sp+Nf);

            const bmm a08 = addpz16(x0, x8); const bmm s08 = subpz16(x0, x8);
            const bmm a4c = addpz16(x4, xc); const bmm s4c = subpz16(x4, xc);
            const bmm a2a = addpz16(x2, xa); const bmm s2a = subpz16(x2, xa);
            const bmm a6e = addpz16(x6, xe); const bmm s6e = subpz16(x6, xe);
            const bmm a19 = addpz16(x1, x9); const bmm s19 = subpz16(x1, x9);
            const bmm a5d = addpz16(x5, xd); const bmm s5d = subpz16(x5, xd);
            const bmm a3b = addpz16(x3, xb); const bmm s3b = subpz16(x3, xb);
            const bmm a7f = addpz16(x7, xf); const bmm s7f = subpz16(x7, xf);

            const bmm js4c = jxpz16(s4c);
            const bmm js6e = jxpz16(s6e);
            const bmm js5d = jxpz16(s5d);
            const bmm js7f = jxpz16(s7f);

            const bmm a08p1a4c = addpz16(a08, a4c); const bmm s08mjs4c = subpz16(s08, js4c);
            const bmm a08m1a4c = subpz16(a08, a4c); const bmm s08pjs4c = addpz16(s08, js4c);
            const bmm a2ap1a6e = addpz16(a2a, a6e); const bmm s2amjs6e = subpz16(s2a, js6e);
            const bmm a2am1a6e = subpz16(a2a, a6e); const bmm s2apjs6e = addpz16(s2a, js6e);
            const bmm a19p1a5d = addpz16(a19, a5d); const bmm s19mjs5d = subpz16(s19, js5d);
            const bmm a19m1a5d = subpz16(a19, a5d); const bmm s19pjs5d = addpz16(s19, js5d);
            const bmm a3bp1a7f = addpz16(a3b, a7f); const bmm s3bmjs7f = subpz16(s3b, js7f);
            const bmm a3bm1a7f = subpz16(a3b, a7f); const bmm s3bpjs7f = addpz16(s3b, js7f);

            const bmm w8_s2amjs6e = w8xpz16(s2amjs6e);
            const bmm  j_a2am1a6e =  jxpz16(a2am1a6e);
            const bmm v8_s2apjs6e = v8xpz16(s2apjs6e);

            const bmm a08p1a4c_p1_a2ap1a6e = addpz16(a08p1a4c,    a2ap1a6e);
            const bmm s08mjs4c_pw_s2amjs6e = addpz16(s08mjs4c, w8_s2amjs6e);
            const bmm a08m1a4c_mj_a2am1a6e = subpz16(a08m1a4c,  j_a2am1a6e);
            const bmm s08pjs4c_mv_s2apjs6e = subpz16(s08pjs4c, v8_s2apjs6e);
            const bmm a08p1a4c_m1_a2ap1a6e = subpz16(a08p1a4c,    a2ap1a6e);
            const bmm s08mjs4c_mw_s2amjs6e = subpz16(s08mjs4c, w8_s2amjs6e);
            const bmm a08m1a4c_pj_a2am1a6e = addpz16(a08m1a4c,  j_a2am1a6e);
            const bmm s08pjs4c_pv_s2apjs6e = addpz16(s08pjs4c, v8_s2apjs6e);

            const bmm w8_s3bmjs7f = w8xpz16(s3bmjs7f);
            const bmm  j_a3bm1a7f =  jxpz16(a3bm1a7f);
            const bmm v8_s3bpjs7f = v8xpz16(s3bpjs7f);

            const bmm a19p1a5d_p1_a3bp1a7f = addpz16(a19p1a5d,    a3bp1a7f);
            const bmm s19mjs5d_pw_s3bmjs7f = addpz16(s19mjs5d, w8_s3bmjs7f);
            const bmm a19m1a5d_mj_a3bm1a7f = subpz16(a19m1a5d,  j_a3bm1a7f);
            const bmm s19pjs5d_mv_s3bpjs7f = subpz16(s19pjs5d, v8_s3bpjs7f);
            const bmm a19p1a5d_m1_a3bp1a7f = subpz16(a19p1a5d,    a3bp1a7f);
            const bmm s19mjs5d_mw_s3bmjs7f = subpz16(s19mjs5d, w8_s3bmjs7f);
            const bmm a19m1a5d_pj_a3bm1a7f = addpz16(a19m1a5d,  j_a3bm1a7f);
            const bmm s19pjs5d_pv_s3bpjs7f = addpz16(s19pjs5d, v8_s3bpjs7f);

            const bmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz16(s19mjs5d_pw_s3bmjs7f);
            const bmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz16(a19m1a5d_mj_a3bm1a7f);
            const bmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz16(s19pjs5d_mv_s3bpjs7f);
            const bmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz16(a19p1a5d_m1_a3bp1a7f);
            const bmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz16(s19mjs5d_mw_s3bmjs7f);
            const bmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz16(a19m1a5d_pj_a3bm1a7f);
            const bmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz16(s19pjs5d_pv_s3bpjs7f);

            const bmm w2p = mulpz16(w1p,w1p);
            const bmm w3p = mulpz16(w1p,w2p);
            const bmm w4p = mulpz16(w2p,w2p);
            const bmm w5p = mulpz16(w2p,w3p);
            const bmm w6p = mulpz16(w3p,w3p);
            const bmm w7p = mulpz16(w3p,w4p);
            const bmm w8p = mulpz16(w4p,w4p);
            const bmm w9p = mulpz16(w4p,w5p);
            const bmm wap = mulpz16(w5p,w5p);
            const bmm wbp = mulpz16(w5p,w6p);
            const bmm wcp = mulpz16(w6p,w6p);
            const bmm wdp = mulpz16(w6p,w7p);
            const bmm wep = mulpz16(w7p,w7p);
            const bmm wfp = mulpz16(w7p,w8p);

            setpz16(yq_s16p+s*0x0,              addpz16(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz16(yq_s16p+s*0x1, mulpz16(w1p, addpz16(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
            setpz16(yq_s16p+s*0x2, mulpz16(w2p, addpz16(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz16(yq_s16p+s*0x3, mulpz16(w3p, addpz16(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz16(yq_s16p+s*0x4, mulpz16(w4p, addpz16(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz16(yq_s16p+s*0x5, mulpz16(w5p, subpz16(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz16(yq_s16p+s*0x6, mulpz16(w6p, subpz16(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz16(yq_s16p+s*0x7, mulpz16(w7p, subpz16(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));

            setpz16(yq_s16p+s*0x8, mulpz16(w8p, subpz16(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
            setpz16(yq_s16p+s*0x9, mulpz16(w9p, subpz16(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
            setpz16(yq_s16p+s*0xa, mulpz16(wap, subpz16(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz16(yq_s16p+s*0xb, mulpz16(wbp, subpz16(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz16(yq_s16p+s*0xc, mulpz16(wcp, subpz16(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz16(yq_s16p+s*0xd, mulpz16(wdp, addpz16(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz16(yq_s16p+s*0xe, mulpz16(wep, addpz16(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz16(yq_s16p+s*0xf, mulpz16(wfp, addpz16(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
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
    static constexpr double theta = 2*M_PI/N;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        #pragma omp for schedule(static)
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p   = x + p;
            complex_vector y_16p = y + 16*p;

            const double t1 = theta*p;
            const double t2 = t1 + theta;
            const ymm w1p = cmplx2(expj(t1), expj(t2));

            const ymm x0  = getpz2(x_p+N0);
            const ymm x1  = getpz2(x_p+N1);
            const ymm x2  = getpz2(x_p+N2);
            const ymm x3  = getpz2(x_p+N3);
            const ymm x4  = getpz2(x_p+N4);
            const ymm x5  = getpz2(x_p+N5);
            const ymm x6  = getpz2(x_p+N6);
            const ymm x7  = getpz2(x_p+N7);
            const ymm x8  = getpz2(x_p+N8);
            const ymm x9  = getpz2(x_p+N9);
            const ymm xa  = getpz2(x_p+Na);
            const ymm xb  = getpz2(x_p+Nb);
            const ymm xc  = getpz2(x_p+Nc);
            const ymm xd  = getpz2(x_p+Nd);
            const ymm xe  = getpz2(x_p+Ne);
            const ymm xf  = getpz2(x_p+Nf);

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
#if 0
            setpz3<16>(y_16p+0x0,             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz3<16>(y_16p+0x1, mulpz2(w1p, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
            setpz3<16>(y_16p+0x2, mulpz2(w2p, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0x3, mulpz2(w3p, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0x4, mulpz2(w4p, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0x5, mulpz2(w5p, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0x6, mulpz2(w6p, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0x7, mulpz2(w7p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));

            setpz3<16>(y_16p+0x8, mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
            setpz3<16>(y_16p+0x9, mulpz2(w9p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
            setpz3<16>(y_16p+0xa, mulpz2(wap, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0xb, mulpz2(wbp, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0xc, mulpz2(wcp, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0xd, mulpz2(wdp, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0xe, mulpz2(wep, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0xf, mulpz2(wfp, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
#else
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
#endif
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
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        invcore<n,s>()(x, y);
        invfft<n/16,16*s,!eo,mode>()(y, x);
    }
};

template <int s, bool eo, int mode> struct invfft<16,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        invend<16,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct invfft<8,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF8omp::invend<8,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct invfft<4,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF4omp::invend<4,s,eo,mode>()(x, y);
    }
};

template <int s, bool eo, int mode> struct invfft<2,s,eo,mode>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        OTFFT_AVXDIF4omp::invend<2,s,eo,mode>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Power of 2 FFT Routine
///////////////////////////////////////////////////////////////////////////////

inline void fwd(const int log_N, complex_vector x, complex_vector y) noexcept
{
    constexpr int mode = scale_length;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

inline void fwd0(const int log_N, complex_vector x, complex_vector y) noexcept
{
    constexpr int mode = scale_1;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

inline void fwdu(const int log_N, complex_vector x, complex_vector y) noexcept
{
    constexpr int mode = scale_unitary;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: fwdfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: fwdfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: fwdfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: fwdfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: fwdfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: fwdfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: fwdfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: fwdfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: fwdfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: fwdfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: fwdfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: fwdfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: fwdfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: fwdfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: fwdfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: fwdfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: fwdfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: fwdfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: fwdfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: fwdfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: fwdfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: fwdfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: fwdfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: fwdfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

inline void fwdn(const int log_N, complex_vector x, complex_vector y) noexcept
{
    fwd(log_N, x, y);
}

///////////////////////////////////////////////////////////////////////////////

inline void inv(const int log_N, complex_vector x, complex_vector y) noexcept
{
    constexpr int mode = scale_1;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

inline void inv0(const int log_N, complex_vector x, complex_vector y) noexcept
{
    inv(log_N, x, y);
}

inline void invu(const int log_N, complex_vector x, complex_vector y) noexcept
{
    constexpr int mode = scale_unitary;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

inline void invn(const int log_N, complex_vector x, complex_vector y) noexcept
{
    constexpr int mode = scale_length;
    #pragma omp parallel
    switch (log_N) {
        case  0: break;
        case  1: invfft<(1<< 1),1,0,mode>()(x, y); break;
        case  2: invfft<(1<< 2),1,0,mode>()(x, y); break;
        case  3: invfft<(1<< 3),1,0,mode>()(x, y); break;
        case  4: invfft<(1<< 4),1,0,mode>()(x, y); break;
        case  5: invfft<(1<< 5),1,0,mode>()(x, y); break;
        case  6: invfft<(1<< 6),1,0,mode>()(x, y); break;
        case  7: invfft<(1<< 7),1,0,mode>()(x, y); break;
        case  8: invfft<(1<< 8),1,0,mode>()(x, y); break;
        case  9: invfft<(1<< 9),1,0,mode>()(x, y); break;
        case 10: invfft<(1<<10),1,0,mode>()(x, y); break;
        case 11: invfft<(1<<11),1,0,mode>()(x, y); break;
        case 12: invfft<(1<<12),1,0,mode>()(x, y); break;
        case 13: invfft<(1<<13),1,0,mode>()(x, y); break;
        case 14: invfft<(1<<14),1,0,mode>()(x, y); break;
        case 15: invfft<(1<<15),1,0,mode>()(x, y); break;
        case 16: invfft<(1<<16),1,0,mode>()(x, y); break;
        case 17: invfft<(1<<17),1,0,mode>()(x, y); break;
        case 18: invfft<(1<<18),1,0,mode>()(x, y); break;
        case 19: invfft<(1<<19),1,0,mode>()(x, y); break;
        case 20: invfft<(1<<20),1,0,mode>()(x, y); break;
        case 21: invfft<(1<<21),1,0,mode>()(x, y); break;
        case 22: invfft<(1<<22),1,0,mode>()(x, y); break;
        case 23: invfft<(1<<23),1,0,mode>()(x, y); break;
        case 24: invfft<(1<<24),1,0,mode>()(x, y); break;
    }
}

} /////////////////////////////////////////////////////////////////////////////

}

#endif // otfft_avxdif16exp_h
