/******************************************************************************
*  OTFFT AVXDIT(Radix-8) Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdit8_h
#define otfft_avxdit8_h

#include "otfft_avxdit4.h"
#include "otfft_avxdit8omp.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_AVXDIT8 { /////////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;

#ifdef DO_SINGLE_THREAD
constexpr int OMP_THRESHOLD = 1<<30;
#else
//constexpr int OMP_THRESHOLD = 1<<15;
constexpr int OMP_THRESHOLD = 1<<13;
#endif

    ///////////////////////////////////////////////////////////////////////////////
    // Forward Buffterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s> struct fwdcore
    {
        static constexpr int m  = n/8;
        static constexpr int N  = n*s;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/8;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s8p = 8*sp;
                const emm w1p = dupez5(*twidT<8,N,1>(W,sp));
                const emm w2p = dupez5(*twidT<8,N,2>(W,sp));
                const emm w3p = dupez5(*twidT<8,N,3>(W,sp));
                const emm w4p = dupez5(*twidT<8,N,4>(W,sp));
                const emm w5p = dupez5(*twidT<8,N,5>(W,sp));
                const emm w6p = dupez5(*twidT<8,N,6>(W,sp));
                const emm w7p = dupez5(*twidT<8,N,7>(W,sp));
                for (int q = 0; q < s; q += 4) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s8p = y + q + s8p;
                    const emm y0 =             getez4(yq_s8p+s*0);
                    const emm y1 = mulez4(w1p, getez4(yq_s8p+s*1));
                    const emm y2 = mulez4(w2p, getez4(yq_s8p+s*2));
                    const emm y3 = mulez4(w3p, getez4(yq_s8p+s*3));
                    const emm y4 = mulez4(w4p, getez4(yq_s8p+s*4));
                    const emm y5 = mulez4(w5p, getez4(yq_s8p+s*5));
                    const emm y6 = mulez4(w6p, getez4(yq_s8p+s*6));
                    const emm y7 = mulez4(w7p, getez4(yq_s8p+s*7));
                    const emm  a04 =       addez4(y0, y4);
                    const emm  s04 =       subez4(y0, y4);
                    const emm  a26 =       addez4(y2, y6);
                    const emm js26 = jxez4(subez4(y2, y6));
                    const emm  a15 =       addez4(y1, y5);
                    const emm  s15 =       subez4(y1, y5);
                    const emm  a37 =       addez4(y3, y7);
                    const emm js37 = jxez4(subez4(y3, y7));
#if 0
                    const emm    a04_p1_a26 =        addez4(a04,  a26);
                    const emm    s04_mj_s26 =        subez4(s04, js26);
                    const emm    a04_m1_a26 =        subez4(a04,  a26);
                    const emm    s04_pj_s26 =        addez4(s04, js26);
                    const emm    a15_p1_a37 =        addez4(a15,  a37);
                    const emm w8_s15_mj_s37 = w8xez4(subez4(s15, js37));
                    const emm  j_a15_m1_a37 =  jxez4(subez4(a15,  a37));
                    const emm v8_s15_pj_s37 = v8xez4(addez4(s15, js37));
                    setez4(xq_sp+N0, addez4(a04_p1_a26,    a15_p1_a37));
                    setez4(xq_sp+N1, addez4(s04_mj_s26, w8_s15_mj_s37));
                    setez4(xq_sp+N2, subez4(a04_m1_a26,  j_a15_m1_a37));
                    setez4(xq_sp+N3, subez4(s04_pj_s26, v8_s15_pj_s37));
                    setez4(xq_sp+N4, subez4(a04_p1_a26,    a15_p1_a37));
                    setez4(xq_sp+N5, subez4(s04_mj_s26, w8_s15_mj_s37));
                    setez4(xq_sp+N6, addez4(a04_m1_a26,  j_a15_m1_a37));
                    setez4(xq_sp+N7, addez4(s04_pj_s26, v8_s15_pj_s37));
#else
                    const emm    a04_p1_a26 =        addez4(a04,  a26);
                    const emm    a15_p1_a37 =        addez4(a15,  a37);
                    setez4(xq_sp+N0, addez4(a04_p1_a26,    a15_p1_a37));
                    setez4(xq_sp+N4, subez4(a04_p1_a26,    a15_p1_a37));

                    const emm    s04_mj_s26 =        subez4(s04, js26);
                    const emm w8_s15_mj_s37 = w8xez4(subez4(s15, js37));
                    setez4(xq_sp+N1, addez4(s04_mj_s26, w8_s15_mj_s37));
                    setez4(xq_sp+N5, subez4(s04_mj_s26, w8_s15_mj_s37));

                    const emm    a04_m1_a26 =        subez4(a04,  a26);
                    const emm  j_a15_m1_a37 =  jxez4(subez4(a15,  a37));
                    setez4(xq_sp+N2, subez4(a04_m1_a26,  j_a15_m1_a37));
                    setez4(xq_sp+N6, addez4(a04_m1_a26,  j_a15_m1_a37));

                    const emm    s04_pj_s26 =        addez4(s04, js26);
                    const emm v8_s15_pj_s37 = v8xez4(addez4(s15, js37));
                    setez4(xq_sp+N3, subez4(s04_pj_s26, v8_s15_pj_s37));
                    setez4(xq_sp+N7, addez4(s04_pj_s26, v8_s15_pj_s37));
#endif
                }
            }
        }
    };

    template <int N> struct fwdcore<N,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = N/8;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
#if 0
                const ymm w1p = getpz2(W+p);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm y0 =             getpz3<8>(y_8p+0);
                const ymm y1 = mulpz2(w1p, getpz3<8>(y_8p+1));
                const ymm y2 = mulpz2(w2p, getpz3<8>(y_8p+2));
                const ymm y3 = mulpz2(w3p, getpz3<8>(y_8p+3));
                const ymm y4 = mulpz2(w4p, getpz3<8>(y_8p+4));
                const ymm y5 = mulpz2(w5p, getpz3<8>(y_8p+5));
                const ymm y6 = mulpz2(w6p, getpz3<8>(y_8p+6));
                const ymm y7 = mulpz2(w7p, getpz3<8>(y_8p+7));
#else
                const ymm w1p = getpz2(twid<8,N,1>(W,p));
                const ymm w2p = getpz2(twid<8,N,2>(W,p));
                const ymm w3p = getpz2(twid<8,N,3>(W,p));
                const ymm w4p = getpz2(twid<8,N,4>(W,p));
                const ymm w5p = getpz2(twid<8,N,5>(W,p));
                const ymm w6p = getpz2(twid<8,N,6>(W,p));
                const ymm w7p = getpz2(twid<8,N,7>(W,p));
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
#endif
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
#if 0
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
#else
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                setpz2(x_p+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(x_p+N4, subpz2(a04_p1_a26,    a15_p1_a37));

                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(x_p+N1, addpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(x_p+N5, subpz2(s04_mj_s26, w8_s15_mj_s37));

                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                setpz2(x_p+N2, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(x_p+N6, addpz2(a04_m1_a26,  j_a15_m1_a37));

                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(x_p+N3, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(x_p+N7, addpz2(s04_pj_s26, v8_s15_pj_s37));
#endif
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct fwdend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct fwdend<8,s,eo,mode>
    {
        static constexpr int N = 8*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm z0 = scalepz2<N,mode>(getpz2(zq+s*0));
                const ymm z1 = scalepz2<N,mode>(getpz2(zq+s*1));
                const ymm z2 = scalepz2<N,mode>(getpz2(zq+s*2));
                const ymm z3 = scalepz2<N,mode>(getpz2(zq+s*3));
                const ymm z4 = scalepz2<N,mode>(getpz2(zq+s*4));
                const ymm z5 = scalepz2<N,mode>(getpz2(zq+s*5));
                const ymm z6 = scalepz2<N,mode>(getpz2(zq+s*6));
                const ymm z7 = scalepz2<N,mode>(getpz2(zq+s*7));
                const ymm  a04 =       addpz2(z0, z4);
                const ymm  s04 =       subpz2(z0, z4);
                const ymm  a26 =       addpz2(z2, z6);
                const ymm js26 = jxpz2(subpz2(z2, z6));
                const ymm  a15 =       addpz2(z1, z5);
                const ymm  s15 =       subpz2(z1, z5);
                const ymm  a37 =       addpz2(z3, z7);
                const ymm js37 = jxpz2(subpz2(z3, z7));
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
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm z0 = scalepz<8,mode>(getpz(z[0]));
            const xmm z1 = scalepz<8,mode>(getpz(z[1]));
            const xmm z2 = scalepz<8,mode>(getpz(z[2]));
            const xmm z3 = scalepz<8,mode>(getpz(z[3]));
            const xmm z4 = scalepz<8,mode>(getpz(z[4]));
            const xmm z5 = scalepz<8,mode>(getpz(z[5]));
            const xmm z6 = scalepz<8,mode>(getpz(z[6]));
            const xmm z7 = scalepz<8,mode>(getpz(z[7]));
            const xmm  a04 =      addpz(z0, z4);
            const xmm  s04 =      subpz(z0, z4);
            const xmm  a26 =      addpz(z2, z6);
            const xmm js26 = jxpz(subpz(z2, z6));
            const xmm  a15 =      addpz(z1, z5);
            const xmm  s15 =      subpz(z1, z5);
            const xmm  a37 =      addpz(z3, z7);
            const xmm js37 = jxpz(subpz(z3, z7));
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
            OTFFT_AVXDIT4::fwdend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT4::fwdend<2,s,eo,mode>()(x, y);
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Inverse Butterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s> struct invcore
    {
        static constexpr int m  = n/8;
        static constexpr int N  = n*s;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/8;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s8p = 8*sp;
#if 0
                const emm w1p = cnjez4(dupez5(*twidT<8,N,1>(W,sp)));
                const emm w2p = cnjez4(dupez5(*twidT<8,N,2>(W,sp)));
                const emm w3p = cnjez4(dupez5(*twidT<8,N,3>(W,sp)));
                const emm w4p = cnjez4(dupez5(*twidT<8,N,4>(W,sp)));
                const emm w5p = cnjez4(dupez5(*twidT<8,N,5>(W,sp)));
                const emm w6p = cnjez4(dupez5(*twidT<8,N,6>(W,sp)));
                const emm w7p = cnjez4(dupez5(*twidT<8,N,7>(W,sp)));
#else
                const emm w1p = dupez5(conj(*twidT<8,N,1>(W,sp)));
                const emm w2p = dupez5(conj(*twidT<8,N,2>(W,sp)));
                const emm w3p = dupez5(conj(*twidT<8,N,3>(W,sp)));
                const emm w4p = dupez5(conj(*twidT<8,N,4>(W,sp)));
                const emm w5p = dupez5(conj(*twidT<8,N,5>(W,sp)));
                const emm w6p = dupez5(conj(*twidT<8,N,6>(W,sp)));
                const emm w7p = dupez5(conj(*twidT<8,N,7>(W,sp)));
#endif
                for (int q = 0; q < s; q += 4) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s8p = y + q + s8p;
                    const emm y0 =             getez4(yq_s8p+s*0);
                    const emm y1 = mulez4(w1p, getez4(yq_s8p+s*1));
                    const emm y2 = mulez4(w2p, getez4(yq_s8p+s*2));
                    const emm y3 = mulez4(w3p, getez4(yq_s8p+s*3));
                    const emm y4 = mulez4(w4p, getez4(yq_s8p+s*4));
                    const emm y5 = mulez4(w5p, getez4(yq_s8p+s*5));
                    const emm y6 = mulez4(w6p, getez4(yq_s8p+s*6));
                    const emm y7 = mulez4(w7p, getez4(yq_s8p+s*7));
                    const emm  a04 =       addez4(y0, y4);
                    const emm  s04 =       subez4(y0, y4);
                    const emm  a26 =       addez4(y2, y6);
                    const emm js26 = jxez4(subez4(y2, y6));
                    const emm  a15 =       addez4(y1, y5);
                    const emm  s15 =       subez4(y1, y5);
                    const emm  a37 =       addez4(y3, y7);
                    const emm js37 = jxez4(subez4(y3, y7));
#if 0
                    const emm    a04_p1_a26 =        addez4(a04,  a26);
                    const emm    s04_pj_s26 =        addez4(s04, js26);
                    const emm    a04_m1_a26 =        subez4(a04,  a26);
                    const emm    s04_mj_s26 =        subez4(s04, js26);
                    const emm    a15_p1_a37 =        addez4(a15,  a37);
                    const emm v8_s15_pj_s37 = v8xez4(addez4(s15, js37));
                    const emm  j_a15_m1_a37 =  jxez4(subez4(a15,  a37));
                    const emm w8_s15_mj_s37 = w8xez4(subez4(s15, js37));
                    setez4(xq_sp+N0, addez4(a04_p1_a26,    a15_p1_a37));
                    setez4(xq_sp+N1, addez4(s04_pj_s26, v8_s15_pj_s37));
                    setez4(xq_sp+N2, addez4(a04_m1_a26,  j_a15_m1_a37));
                    setez4(xq_sp+N3, subez4(s04_mj_s26, w8_s15_mj_s37));
                    setez4(xq_sp+N4, subez4(a04_p1_a26,    a15_p1_a37));
                    setez4(xq_sp+N5, subez4(s04_pj_s26, v8_s15_pj_s37));
                    setez4(xq_sp+N6, subez4(a04_m1_a26,  j_a15_m1_a37));
                    setez4(xq_sp+N7, addez4(s04_mj_s26, w8_s15_mj_s37));
#else
                    const emm    a04_p1_a26 =        addez4(a04,  a26);
                    const emm    a15_p1_a37 =        addez4(a15,  a37);
                    setez4(xq_sp+N0, addez4(a04_p1_a26,    a15_p1_a37));
                    setez4(xq_sp+N4, subez4(a04_p1_a26,    a15_p1_a37));

                    const emm    s04_pj_s26 =        addez4(s04, js26);
                    const emm v8_s15_pj_s37 = v8xez4(addez4(s15, js37));
                    setez4(xq_sp+N1, addez4(s04_pj_s26, v8_s15_pj_s37));
                    setez4(xq_sp+N5, subez4(s04_pj_s26, v8_s15_pj_s37));

                    const emm    a04_m1_a26 =        subez4(a04,  a26);
                    const emm  j_a15_m1_a37 =  jxez4(subez4(a15,  a37));
                    setez4(xq_sp+N2, addez4(a04_m1_a26,  j_a15_m1_a37));
                    setez4(xq_sp+N6, subez4(a04_m1_a26,  j_a15_m1_a37));

                    const emm    s04_mj_s26 =        subez4(s04, js26);
                    const emm w8_s15_mj_s37 = w8xez4(subez4(s15, js37));
                    setez4(xq_sp+N3, subez4(s04_mj_s26, w8_s15_mj_s37));
                    setez4(xq_sp+N7, addez4(s04_mj_s26, w8_s15_mj_s37));
#endif
                }
            }
        }
    };

    template <int N> struct invcore<N,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = N/8;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < N/8; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
#if 0
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm y0 =             getpz3<8>(y_8p+0);
                const ymm y1 = mulpz2(w1p, getpz3<8>(y_8p+1));
                const ymm y2 = mulpz2(w2p, getpz3<8>(y_8p+2));
                const ymm y3 = mulpz2(w3p, getpz3<8>(y_8p+3));
                const ymm y4 = mulpz2(w4p, getpz3<8>(y_8p+4));
                const ymm y5 = mulpz2(w5p, getpz3<8>(y_8p+5));
                const ymm y6 = mulpz2(w6p, getpz3<8>(y_8p+6));
                const ymm y7 = mulpz2(w7p, getpz3<8>(y_8p+7));
#else
                const ymm w1p = cnjpz2(getpz2(twid<8,N,1>(W,p)));
                const ymm w2p = cnjpz2(getpz2(twid<8,N,2>(W,p)));
                const ymm w3p = cnjpz2(getpz2(twid<8,N,3>(W,p)));
                const ymm w4p = cnjpz2(getpz2(twid<8,N,4>(W,p)));
                const ymm w5p = cnjpz2(getpz2(twid<8,N,5>(W,p)));
                const ymm w6p = cnjpz2(getpz2(twid<8,N,6>(W,p)));
                const ymm w7p = cnjpz2(getpz2(twid<8,N,7>(W,p)));
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
#endif
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
#if 0
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
#else
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                setpz2(x_p+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(x_p+N4, subpz2(a04_p1_a26,    a15_p1_a37));

                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(x_p+N1, addpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(x_p+N5, subpz2(s04_pj_s26, v8_s15_pj_s37));

                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                setpz2(x_p+N2, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(x_p+N6, subpz2(a04_m1_a26,  j_a15_m1_a37));

                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(x_p+N3, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(x_p+N7, addpz2(s04_mj_s26, w8_s15_mj_s37));
#endif
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct invend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct invend<8,s,eo,mode>
    {
        static constexpr int N  = 8*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm z0 = scalepz2<N,mode>(getpz2(zq+s*0));
                const ymm z1 = scalepz2<N,mode>(getpz2(zq+s*1));
                const ymm z2 = scalepz2<N,mode>(getpz2(zq+s*2));
                const ymm z3 = scalepz2<N,mode>(getpz2(zq+s*3));
                const ymm z4 = scalepz2<N,mode>(getpz2(zq+s*4));
                const ymm z5 = scalepz2<N,mode>(getpz2(zq+s*5));
                const ymm z6 = scalepz2<N,mode>(getpz2(zq+s*6));
                const ymm z7 = scalepz2<N,mode>(getpz2(zq+s*7));
                const ymm  a04 =       addpz2(z0, z4);
                const ymm  s04 =       subpz2(z0, z4);
                const ymm  a26 =       addpz2(z2, z6);
                const ymm js26 = jxpz2(subpz2(z2, z6));
                const ymm  a15 =       addpz2(z1, z5);
                const ymm  s15 =       subpz2(z1, z5);
                const ymm  a37 =       addpz2(z3, z7);
                const ymm js37 = jxpz2(subpz2(z3, z7));
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
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm z0 = scalepz<8,mode>(getpz(z[0]));
            const xmm z1 = scalepz<8,mode>(getpz(z[1]));
            const xmm z2 = scalepz<8,mode>(getpz(z[2]));
            const xmm z3 = scalepz<8,mode>(getpz(z[3]));
            const xmm z4 = scalepz<8,mode>(getpz(z[4]));
            const xmm z5 = scalepz<8,mode>(getpz(z[5]));
            const xmm z6 = scalepz<8,mode>(getpz(z[6]));
            const xmm z7 = scalepz<8,mode>(getpz(z[7]));
            const xmm  a04 =      addpz(z0, z4);
            const xmm  s04 =      subpz(z0, z4);
            const xmm  a26 =      addpz(z2, z6);
            const xmm js26 = jxpz(subpz(z2, z6));
            const xmm  a15 =      addpz(z1, z5);
            const xmm  s15 =      subpz(z1, z5);
            const xmm  a37 =      addpz(z3, z7);
            const xmm js37 = jxpz(subpz(z3, z7));
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
            OTFFT_AVXDIT4::invend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct invfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            OTFFT_AVXDIT4::invend<2,s,eo,mode>()(x, y);
        }
    };

} /////////////////////////////////////////////////////////////////////////////

}

#endif // otfft_avxdit8_h
