/******************************************************************************
*  OTFFT AVXDIT(Radix-4) Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdit4_h
#define otfft_avxdit4_h

#include "otfft_avxdit4omp.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_AVXDIT4 { /////////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;

#ifdef DO_SINGLE_THREAD
    constexpr int OMP_THRESHOLD = 1<<30;
#else
    //constexpr int OMP_THRESHOLD = 1<<15;
    constexpr int OMP_THRESHOLD = 1<<13;
#endif

    ///////////////////////////////////////////////////////////////////////////////
    // Forward Butterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s> struct fwdcore
    {
        static constexpr int m  = n/4;
        static constexpr int N  = n*s;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/4;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
                const emm w1p = dupez5(*twidT<4,N,1>(W,sp));
                const emm w2p = dupez5(*twidT<4,N,2>(W,sp));
                const emm w3p = dupez5(*twidT<4,N,3>(W,sp));
                for (int q = 0; q < s; q += 4) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const emm a =             getez4(yq_s4p+s*0);
                    const emm b = mulez4(w1p, getez4(yq_s4p+s*1));
                    const emm c = mulez4(w2p, getez4(yq_s4p+s*2));
                    const emm d = mulez4(w3p, getez4(yq_s4p+s*3));
                    const emm  apc =       addez4(a, c);
                    const emm  amc =       subez4(a, c);
                    const emm  bpd =       addez4(b, d);
                    const emm jbmd = jxez4(subez4(b, d));
                    setez4(xq_sp+N0, addez4(apc,  bpd));
                    setez4(xq_sp+N1, subez4(amc, jbmd));
                    setez4(xq_sp+N2, subez4(apc,  bpd));
                    setez4(xq_sp+N3, addez4(amc, jbmd));
                }
            }
        }
    };

    template <int N> struct fwdcore<N,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = N/4;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
#if 0
                const ymm w1p = getpz2(W+p);
                const ymm w2p = mulpz2(w1p,w1p);
                const ymm w3p = mulpz2(w1p,w2p);
                const ymm a =             getpz3<4>(y_4p+0);
                const ymm b = mulpz2(w1p, getpz3<4>(y_4p+1));
                const ymm c = mulpz2(w2p, getpz3<4>(y_4p+2));
                const ymm d = mulpz2(w3p, getpz3<4>(y_4p+3));
#else
                const ymm w1p = getpz2(twid<4,N,1>(W,p));
                const ymm w2p = getpz2(twid<4,N,2>(W,p));
                const ymm w3p = getpz2(twid<4,N,3>(W,p));
                const ymm ab = getpz2(y_4p+0);
                const ymm cd = getpz2(y_4p+2);
                const ymm AB  = getpz2(y_4p+4);
                const ymm CD  = getpz2(y_4p+6);
                const ymm a =             catlo(ab, AB);
                const ymm b = mulpz2(w1p, cathi(ab, AB));
                const ymm c = mulpz2(w2p, catlo(cd, CD));
                const ymm d = mulpz2(w3p, cathi(cd, CD));
#endif
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x_p+N0, addpz2(apc,  bpd));
                setpz2(x_p+N1, subpz2(amc, jbmd));
                setpz2(x_p+N2, subpz2(apc,  bpd));
                setpz2(x_p+N3, addpz2(amc, jbmd));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct fwdend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct fwdend<4,s,eo,mode>
    {
        static constexpr int N = 4*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = scalepz2<N,mode>(getpz2(zq+s*0));
                const ymm b = scalepz2<N,mode>(getpz2(zq+s*1));
                const ymm c = scalepz2<N,mode>(getpz2(zq+s*2));
                const ymm d = scalepz2<N,mode>(getpz2(zq+s*3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq+s*0, addpz2(apc,  bpd));
                setpz2(xq+s*1, subpz2(amc, jbmd));
                setpz2(xq+s*2, subpz2(apc,  bpd));
                setpz2(xq+s*3, addpz2(amc, jbmd));
            }
        }
    };

    template <bool eo, int mode> struct fwdend<4,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<4,mode>(getpz(z[0]));
            const xmm b = scalepz<4,mode>(getpz(z[1]));
            const xmm c = scalepz<4,mode>(getpz(z[2]));
            const xmm d = scalepz<4,mode>(getpz(z[3]));
            const xmm  apc =      addpz(a, c);
            const xmm  amc =      subpz(a, c);
            const xmm  bpd =      addpz(b, d);
            const xmm jbmd = jxpz(subpz(b, d));
            setpz(x[0], addpz(apc,  bpd));
            setpz(x[1], subpz(amc, jbmd));
            setpz(x[2], subpz(apc,  bpd));
            setpz(x[3], addpz(amc, jbmd));
        }
    };

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct fwdend<2,s,eo,mode>
    {
        static constexpr int N = 2*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = scalepz2<N,mode>(getpz2(zq+0));
                const ymm b = scalepz2<N,mode>(getpz2(zq+s));
                setpz2(xq+0, addpz2(a, b));
                setpz2(xq+s, subpz2(a, b));
            }
        }
    };

    template <bool eo, int mode> struct fwdend<2,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<2,mode>(getpz(z[0]));
            const xmm b = scalepz<2,mode>(getpz(z[1]));
            setpz(x[0], addpz(a, b));
            setpz(x[1], subpz(a, b));
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
            fwdfft<n/4,4*s,!eo,mode>()(y, x, W);
            fwdcore<n,s>()(x, y, W);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<4,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            fwdend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct fwdfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            fwdend<2,s,eo,mode>()(x, y);
        }
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Inverse Butterfly Operation
    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s> struct invcore
    {
        static constexpr int m  = n/4;
        static constexpr int N  = n*s;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/4;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < m; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
#if 0
                const emm w1p = cnjez4(dupez5(*twidT<4,N,1>(W,sp)));
                const emm w2p = cnjez4(dupez5(*twidT<4,N,2>(W,sp)));
                const emm w3p = cnjez4(dupez5(*twidT<4,N,3>(W,sp)));
#else
                const emm w1p = dupez5(conj(*twidT<4,N,1>(W,sp)));
                const emm w2p = dupez5(conj(*twidT<4,N,2>(W,sp)));
                const emm w3p = dupez5(conj(*twidT<4,N,3>(W,sp)));
#endif
                for (int q = 0; q < s; q += 4) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const emm a =             getez4(yq_s4p+s*0);
                    const emm b = mulez4(w1p, getez4(yq_s4p+s*1));
                    const emm c = mulez4(w2p, getez4(yq_s4p+s*2));
                    const emm d = mulez4(w3p, getez4(yq_s4p+s*3));
                    const emm  apc =       addez4(a, c);
                    const emm  amc =       subez4(a, c);
                    const emm  bpd =       addez4(b, d);
                    const emm jbmd = jxez4(subez4(b, d));
                    setez4(xq_sp+N0, addez4(apc,  bpd));
                    setez4(xq_sp+N1, addez4(amc, jbmd));
                    setez4(xq_sp+N2, subez4(apc,  bpd));
                    setez4(xq_sp+N3, subez4(amc, jbmd));
                }
            }
        }
    };

    template <int N> struct invcore<N,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = N/4;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
#if 0
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = mulpz2(w1p,w1p);
                const ymm w3p = mulpz2(w1p,w2p);
                const ymm a =             getpz3<4>(y_4p+0);
                const ymm b = mulpz2(w1p, getpz3<4>(y_4p+1));
                const ymm c = mulpz2(w2p, getpz3<4>(y_4p+2));
                const ymm d = mulpz2(w3p, getpz3<4>(y_4p+3));
#else
                const ymm w1p = cnjpz2(getpz2(twid<4,N,1>(W,p)));
                const ymm w2p = cnjpz2(getpz2(twid<4,N,2>(W,p)));
                const ymm w3p = cnjpz2(getpz2(twid<4,N,3>(W,p)));
                const ymm ab = getpz2(y_4p+0);
                const ymm cd = getpz2(y_4p+2);
                const ymm AB  = getpz2(y_4p+4);
                const ymm CD  = getpz2(y_4p+6);
                const ymm a =             catlo(ab, AB);
                const ymm b = mulpz2(w1p, cathi(ab, AB));
                const ymm c = mulpz2(w2p, catlo(cd, CD));
                const ymm d = mulpz2(w3p, cathi(cd, CD));
#endif
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x_p+N0, addpz2(apc,  bpd));
                setpz2(x_p+N1, addpz2(amc, jbmd));
                setpz2(x_p+N2, subpz2(apc,  bpd));
                setpz2(x_p+N3, subpz2(amc, jbmd));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct invend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct invend<4,s,eo,mode>
    {
        static constexpr int N  = 4*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = scalepz2<N,mode>(getpz2(zq+s*0));
                const ymm b = scalepz2<N,mode>(getpz2(zq+s*1));
                const ymm c = scalepz2<N,mode>(getpz2(zq+s*2));
                const ymm d = scalepz2<N,mode>(getpz2(zq+s*3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq+s*0, addpz2(apc,  bpd));
                setpz2(xq+s*1, addpz2(amc, jbmd));
                setpz2(xq+s*2, subpz2(apc,  bpd));
                setpz2(xq+s*3, subpz2(amc, jbmd));
            }
        }
    };

    template <bool eo, int mode> struct invend<4,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<4,mode>(getpz(z[0]));
            const xmm b = scalepz<4,mode>(getpz(z[1]));
            const xmm c = scalepz<4,mode>(getpz(z[2]));
            const xmm d = scalepz<4,mode>(getpz(z[3]));
            const xmm  apc =      addpz(a, c);
            const xmm  amc =      subpz(a, c);
            const xmm  bpd =      addpz(b, d);
            const xmm jbmd = jxpz(subpz(b, d));
            setpz(x[0], addpz(apc,  bpd));
            setpz(x[1], addpz(amc, jbmd));
            setpz(x[2], subpz(apc,  bpd));
            setpz(x[3], subpz(amc, jbmd));
        }
    };

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct invend<2,s,eo,mode>
    {
        static constexpr int N  = 2*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = scalepz2<N,mode>(getpz2(zq+0));
                const ymm b = scalepz2<N,mode>(getpz2(zq+s));
                setpz2(xq+0, addpz2(a, b));
                setpz2(xq+s, subpz2(a, b));
            }
        }
    };

    template <bool eo, int mode> struct invend<2,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<2,mode>(getpz(z[0]));
            const xmm b = scalepz<2,mode>(getpz(z[1]));
            setpz(x[0], addpz(a, b));
            setpz(x[1], subpz(a, b));
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
            invfft<n/4,4*s,!eo,mode>()(y, x, W);
            invcore<n,s>()(x, y, W);
        }
    };

    template <int s, bool eo, int mode> struct invfft<4,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            invend<4,s,eo,mode>()(x, y);
        }
    };

    template <int s, bool eo, int mode> struct invfft<2,s,eo,mode>
    {
        inline void operator()(
                complex_vector x, complex_vector y, const_complex_vector) const noexcept
        {
            invend<2,s,eo,mode>()(x, y);
        }
    };

} /////////////////////////////////////////////////////////////////////////////

}

#endif // otfft_avxdit4_h
