// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

#include "otfft_misc.h"
#include "otfft_avxdif4omp.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_AVXDIF4 { /////////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;

    constexpr int OMP_THRESHOLD = 1<<15;
    constexpr int AVX_THRESHOLD = 1<<10;

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
                const ymm w1p = getpz2(W+p);
                const ymm w2p = getwp2<2>(W,p);
                const ymm w3p = getwp2<3>(W,p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                if (N < AVX_THRESHOLD) {
                    setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                    setpz3<4>(y_4p+1, mulpz2(w1p, subpz2(amc, jbmd)));
                    setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz3<4>(y_4p+3, mulpz2(w3p, addpz2(amc, jbmd)));
                }
                else {
                    const ymm ab =             addpz2(apc,  bpd);
                    const ymm cd = mulpz2(w1p, subpz2(amc, jbmd));
                    const ymm ef = mulpz2(w2p, subpz2(apc,  bpd));
                    const ymm gh = mulpz2(w3p, addpz2(amc, jbmd));
                    const ymm ac = catlo(ab, cd);
                    const ymm bd = cathi(ab, cd);
                    const ymm eg = catlo(ef, gh);
                    const ymm fh = cathi(ef, gh);
                    setpz2(y_4p+0, ac);
                    setpz2(y_4p+2, eg);
                    setpz2(y_4p+4, bd);
                    setpz2(y_4p+6, fh);
                }
            }
        }
    };

    template <> struct fwdcore<1024,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = 1024/4;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                const ymm w1p = getpz2(W+p);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                const ymm ab =             addpz2(apc,  bpd);
                const ymm cd = mulpz2(w1p, subpz2(amc, jbmd));
                const ymm ef = mulpz2(w2p, subpz2(apc,  bpd));
                const ymm gh = mulpz2(w3p, addpz2(amc, jbmd));
                const ymm ac = catlo(ab, cd);
                const ymm bd = cathi(ab, cd);
                const ymm eg = catlo(ef, gh);
                const ymm fh = cathi(ef, gh);
                setpz2(y_4p+0, ac);
                setpz2(y_4p+2, eg);
                setpz2(y_4p+4, bd);
                setpz2(y_4p+6, fh);
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
                const ymm a = scalepz2<N,mode>(getpz2(xq+s*0));
                const ymm b = scalepz2<N,mode>(getpz2(xq+s*1));
                const ymm c = scalepz2<N,mode>(getpz2(xq+s*2));
                const ymm d = scalepz2<N,mode>(getpz2(xq+s*3));
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
    };

    template <bool eo, int mode> struct fwdend<4,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<4,mode>(getpz(x[0]));
            const xmm b = scalepz<4,mode>(getpz(x[1]));
            const xmm c = scalepz<4,mode>(getpz(x[2]));
            const xmm d = scalepz<4,mode>(getpz(x[3]));
            const xmm  apc =      addpz(a, c);
            const xmm  amc =      subpz(a, c);
            const xmm  bpd =      addpz(b, d);
            const xmm jbmd = jxpz(subpz(b, d));
            setpz(z[0], addpz(apc,  bpd));
            setpz(z[1], subpz(amc, jbmd));
            setpz(z[2], subpz(apc,  bpd));
            setpz(z[3], addpz(amc, jbmd));
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
                const ymm a = scalepz2<N,mode>(getpz2(xq+0));
                const ymm b = scalepz2<N,mode>(getpz2(xq+s));
                setpz2(zq+0, addpz2(a, b));
                setpz2(zq+s, subpz2(a, b));
            }
        }
    };

    template <bool eo, int mode> struct fwdend<2,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<2,mode>(getpz(x[0]));
            const xmm b = scalepz<2,mode>(getpz(x[1]));
            setpz(z[0], addpz(a, b));
            setpz(z[1], subpz(a, b));
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
            fwdfft<n/4,4*s,!eo,mode>()(y, x, W);
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
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = getwp2<-2>(W+N,p);
                const ymm w3p = getwp2<-3>(W+N,p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                if (N < AVX_THRESHOLD) {
                    setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                    setpz3<4>(y_4p+1, mulpz2(w1p, addpz2(amc, jbmd)));
                    setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz3<4>(y_4p+3, mulpz2(w3p, subpz2(amc, jbmd)));
                }
                else {
                    const ymm ab =             addpz2(apc,  bpd);
                    const ymm cd = mulpz2(w1p, addpz2(amc, jbmd));
                    const ymm ef = mulpz2(w2p, subpz2(apc,  bpd));
                    const ymm gh = mulpz2(w3p, subpz2(amc, jbmd));
                    const ymm ac = catlo(ab, cd);
                    const ymm bd = cathi(ab, cd);
                    const ymm eg = catlo(ef, gh);
                    const ymm fh = cathi(ef, gh);
                    setpz2(y_4p+0, ac);
                    setpz2(y_4p+2, eg);
                    setpz2(y_4p+4, bd);
                    setpz2(y_4p+6, fh);
                }
            }
        }
    };

    template <> struct invcore<1024,1>
    {
        static constexpr int N0 = 0;
        static constexpr int N1 = 1024/4;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;

        void operator()(
                complex_vector x, complex_vector y, const_complex_vector W) const noexcept
        {
            for (int p = 0; p < N1; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                const ymm ab =             addpz2(apc,  bpd);
                const ymm cd = mulpz2(w1p, addpz2(amc, jbmd));
                const ymm ef = mulpz2(w2p, subpz2(apc,  bpd));
                const ymm gh = mulpz2(w3p, subpz2(amc, jbmd));
                const ymm ac = catlo(ab, cd);
                const ymm bd = cathi(ab, cd);
                const ymm eg = catlo(ef, gh);
                const ymm fh = cathi(ef, gh);
                setpz2(y_4p+0, ac);
                setpz2(y_4p+2, eg);
                setpz2(y_4p+4, bd);
                setpz2(y_4p+6, fh);
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int n, int s, bool eo, int mode> struct invend;

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct invend<4,s,eo,mode>
    {
        static constexpr int N = 4*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = scalepz2<N,mode>(getpz2(xq+s*0));
                const ymm b = scalepz2<N,mode>(getpz2(xq+s*1));
                const ymm c = scalepz2<N,mode>(getpz2(xq+s*2));
                const ymm d = scalepz2<N,mode>(getpz2(xq+s*3));
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
    };

    template <bool eo, int mode> struct invend<4,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<4,mode>(getpz(x[0]));
            const xmm b = scalepz<4,mode>(getpz(x[1]));
            const xmm c = scalepz<4,mode>(getpz(x[2]));
            const xmm d = scalepz<4,mode>(getpz(x[3]));
            const xmm  apc =      addpz(a, c);
            const xmm  amc =      subpz(a, c);
            const xmm  bpd =      addpz(b, d);
            const xmm jbmd = jxpz(subpz(b, d));
            setpz(z[0], addpz(apc,  bpd));
            setpz(z[1], addpz(amc, jbmd));
            setpz(z[2], subpz(apc,  bpd));
            setpz(z[3], subpz(amc, jbmd));
        }
    };

    //-----------------------------------------------------------------------------

    template <int s, bool eo, int mode> struct invend<2,s,eo,mode>
    {
        static constexpr int N = 2*s;

        void operator()(complex_vector x, complex_vector y) const noexcept
        {
            complex_vector z = eo ? y : x;
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector zq = z + q;
                const ymm a = scalepz2<N,mode>(getpz2(xq+0));
                const ymm b = scalepz2<N,mode>(getpz2(xq+s));
                setpz2(zq+0, addpz2(a, b));
                setpz2(zq+s, subpz2(a, b));
            }
        }
    };

    template <bool eo, int mode> struct invend<2,1,eo,mode>
    {
        inline void operator()(complex_vector x, complex_vector y) const noexcept
        {
            zeroupper();
            complex_vector z = eo ? y : x;
            const xmm a = scalepz<2,mode>(getpz(x[0]));
            const xmm b = scalepz<2,mode>(getpz(x[1]));
            setpz(z[0], addpz(a, b));
            setpz(z[1], subpz(a, b));
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
            invfft<n/4,4*s,!eo,mode>()(y, x, W);
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
