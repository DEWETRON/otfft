/******************************************************************************
*  OTFFT EightStep Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_eightstep_h
#define otfft_eightstep_h

namespace OTFFT_NAMESPACE {

namespace OTFFT_EightStep { ///////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;
    using OTFFT_SixStep::weight_t;
    using OTFFT_SixStep::const_index_vector;

#ifdef DO_SINGLE_THREAD
constexpr int OMP_THRESHOLD1 = 1<<30;
constexpr int OMP_THRESHOLD2 = 1<<30;
#else
constexpr int OMP_THRESHOLD1 = 1<<13;
constexpr int OMP_THRESHOLD2 = 1<<19;
#endif

    ///////////////////////////////////////////////////////////////////////////////

    template <int log_N, int mode> struct fwdfftr
    {
        static constexpr int N = 1 << log_N;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/8;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;

        static inline void transpose_kernel(
                const int p, complex_vector x, complex_vector y) noexcept
        {
#if 0
            const ymm aA = getpz2(x+p+N0);
            const ymm bB = getpz2(x+p+N1);
            const ymm ab = catlo(aA, bB);
            const ymm AB = cathi(aA, bB);
            setpz2(y+8*p+ 0, ab);
            setpz2(y+8*p+ 8, AB);
            const ymm cC = getpz2(x+p+N2);
            const ymm dD = getpz2(x+p+N3);
            const ymm cd = catlo(cC, dD);
            const ymm CD = cathi(cC, dD);
            setpz2(y+8*p+ 2, cd);
            setpz2(y+8*p+10, CD);
            const ymm eE = getpz2(x+p+N4);
            const ymm fF = getpz2(x+p+N5);
            const ymm ef = catlo(eE, fF);
            const ymm EF = cathi(eE, fF);
            setpz2(y+8*p+ 4, ef);
            setpz2(y+8*p+12, EF);
            const ymm gG = getpz2(x+p+N6);
            const ymm hH = getpz2(x+p+N7);
            const ymm gh = catlo(gG, hH);
            const ymm GH = cathi(gG, hH);
            setpz2(y+8*p+ 6, gh);
            setpz2(y+8*p+14, GH);
#else
        for (int q = 0; q < 8; q += 2) {
            const ymm aA = getpz2(x+p+q*N1+N0);
            const ymm bB = getpz2(x+p+q*N1+N1);
            const ymm ab = catlo(aA, bB);
            const ymm AB = cathi(aA, bB);
            setpz2(y+8*p+q+0, ab);
            setpz2(y+8*p+q+8, AB);
        }
#endif
        }

        static inline void fft_and_mult_twiddle_factor_kernel(
                const int p, complex_vector x, complex_vector y, weight_t W) noexcept
        {
            const ymm x0 = scalepz2<N,mode>(getpz2(x+p+N0));
            const ymm x1 = scalepz2<N,mode>(getpz2(x+p+N1));
            const ymm x2 = scalepz2<N,mode>(getpz2(x+p+N2));
            const ymm x3 = scalepz2<N,mode>(getpz2(x+p+N3));
            const ymm x4 = scalepz2<N,mode>(getpz2(x+p+N4));
            const ymm x5 = scalepz2<N,mode>(getpz2(x+p+N5));
            const ymm x6 = scalepz2<N,mode>(getpz2(x+p+N6));
            const ymm x7 = scalepz2<N,mode>(getpz2(x+p+N7));

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

            const ymm w1p = getpz2(W+p);
            const ymm w2p = mulpz2(w1p, w1p);
            const ymm w3p = mulpz2(w1p, w2p);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);

            setpz2(y+p+N0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(y+p+N1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz2(y+p+N2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(y+p+N3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz2(y+p+N4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz2(y+p+N5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz2(y+p+N6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(y+p+N7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
        }

        void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            if (N < OMP_THRESHOLD1) {
                for (int p = 0; p < N1; p += 2) {
                    fft_and_mult_twiddle_factor_kernel(p, x, y, W);
                }
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N0, x+N0, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N1, x+N1, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N2, x+N2, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N3, x+N3, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N4, x+N4, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N5, x+N5, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N6, x+N6, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+N7, x+N7, W, Ws);
                for (int p = 0; p < N1; p += 2) {
                    transpose_kernel(p, y, x);
                }
            }
            else if (N < OMP_THRESHOLD2)
            #pragma omp parallel firstprivate(ip,x,y,W,Ws)
            {
                #pragma omp for schedule(static)
                for (int p = 0; p < N1; p += 2) {
                    fft_and_mult_twiddle_factor_kernel(p, x, y, W);
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < 8; i++) {
                    const int iN1 = i*N1;
                    OTFFT_SixStep::fwdffts8<log_N-3,scale_1,1>()(ip, y+iN1, x+iN1, W, Ws);
                }
                #pragma omp for schedule(static) nowait
                for (int p = 0; p < N1; p += 2) {
                    transpose_kernel(p, y, x);
                }
            }
            else {
                #pragma omp parallel for schedule(static) firstprivate(x,y,W)
                for (int p = 0; p < N1; p += 2) {
                    fft_and_mult_twiddle_factor_kernel(p, x, y, W);
                }
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N0, x+N0, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N1, x+N1, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N2, x+N2, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N3, x+N3, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N4, x+N4, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N5, x+N5, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N6, x+N6, W, Ws);
                OTFFT_SixStep::fwdffts8<log_N-3,scale_1>()(ip, y+N7, x+N7, W, Ws);
                #pragma omp parallel for schedule(static) firstprivate(x,y)
                for (int p = 0; p < N1; p += 2) {
                    transpose_kernel(p, y, x);
                }
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int log_N, int mode> struct invfftr
    {
        static constexpr int N = 1 << log_N;
        static constexpr int N0 = 0;
        static constexpr int N1 = N/8;
        static constexpr int N2 = N1*2;
        static constexpr int N3 = N1*3;
        static constexpr int N4 = N1*4;
        static constexpr int N5 = N1*5;
        static constexpr int N6 = N1*6;
        static constexpr int N7 = N1*7;

        static inline void transpose_kernel(
                const int p, complex_vector x, complex_vector y) noexcept
        {
            fwdfftr<log_N,mode>::transpose_kernel(p, x, y);
        }

        static inline void fft_and_mult_twiddle_factor_kernel(
                const int p, complex_vector x, complex_vector y, weight_t W) noexcept
        {
            const ymm x0 = scalepz2<N,mode>(getpz2(x+p+N0));
            const ymm x1 = scalepz2<N,mode>(getpz2(x+p+N1));
            const ymm x2 = scalepz2<N,mode>(getpz2(x+p+N2));
            const ymm x3 = scalepz2<N,mode>(getpz2(x+p+N3));
            const ymm x4 = scalepz2<N,mode>(getpz2(x+p+N4));
            const ymm x5 = scalepz2<N,mode>(getpz2(x+p+N5));
            const ymm x6 = scalepz2<N,mode>(getpz2(x+p+N6));
            const ymm x7 = scalepz2<N,mode>(getpz2(x+p+N7));

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

            const ymm w1p = cnjpz2(getpz2(W+p));
            const ymm w2p = mulpz2(w1p,w1p);
            const ymm w3p = mulpz2(w1p,w2p);
            const ymm w4p = mulpz2(w2p,w2p);
            const ymm w5p = mulpz2(w2p,w3p);
            const ymm w6p = mulpz2(w3p,w3p);
            const ymm w7p = mulpz2(w3p,w4p);

            setpz2(y+p+N0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(y+p+N1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz2(y+p+N2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(y+p+N3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz2(y+p+N4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz2(y+p+N5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz2(y+p+N6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(y+p+N7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
        }

        void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            if (N < OMP_THRESHOLD1) {
                for (int p = 0; p < N1; p += 2) {
                    fft_and_mult_twiddle_factor_kernel(p, x, y, W);
                }
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N0, x+N0, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N1, x+N1, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N2, x+N2, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N3, x+N3, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N4, x+N4, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N5, x+N5, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N6, x+N6, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+N7, x+N7, W, Ws);
                for (int p = 0; p < N1; p += 2) {
                    transpose_kernel(p, y, x);
                }
            }
            else if (N < OMP_THRESHOLD2)
            #pragma omp parallel firstprivate(ip, x,y,W,Ws)
            {
                #pragma omp for schedule(static)
                for (int p = 0; p < N1; p += 2) {
                    fft_and_mult_twiddle_factor_kernel(p, x, y, W);
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < 8; i++) {
                    const int iN1 = i*N1;
                    OTFFT_SixStep::invffts8<log_N-3,scale_1,1>()(ip, y+iN1, x+iN1, W, Ws);
                }
                #pragma omp for schedule(static) nowait
                for (int p = 0; p < N1; p += 2) {
                    transpose_kernel(p, y, x);
                }
            }
            else {
                #pragma omp parallel for schedule(static) firstprivate(x,y,W)
                for (int p = 0; p < N1; p += 2) {
                    fft_and_mult_twiddle_factor_kernel(p, x, y, W);
                }
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N0, x+N0, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N1, x+N1, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N2, x+N2, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N3, x+N3, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N4, x+N4, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N5, x+N5, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N6, x+N6, W, Ws);
                OTFFT_SixStep::invffts8<log_N-3,scale_1>()(ip, y+N7, x+N7, W, Ws);
                #pragma omp parallel for schedule(static) firstprivate(x,y)
                for (int p = 0; p < N1; p += 2) {
                    transpose_kernel(p, y, x);
                }
            }
        }
    };

} /////////////////////////////////////////////////////////////////////////////

}

#endif // otfft_eightstep_h
