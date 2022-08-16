/******************************************************************************
*  OTFFT SixStep of Square Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_sixstepsq_h
#define otfft_sixstepsq_h

#include "otfft_avxdif16.h"

namespace OTFFT_NAMESPACE {
namespace OTFFT_FourStep = OTFFT_AVXDIF16;

namespace OTFFT_SixStep { /////////////////////////////////////////////////////

#ifdef DO_SINGLE_THREAD
    constexpr int OMP_THRESHOLD1 = 1<<30;
    constexpr int OMP_THRESHOLD2 = 1<<30;
#else
    constexpr int OMP_THRESHOLD1 = 1<<13;
    constexpr int OMP_THRESHOLD2 = 1<<17;
#endif

    template <int log_N, int s, int mode, bool sng> struct fwdffts_body
    {
        static constexpr int log_n = log_N/2;
        static constexpr int N = 1 << log_N;
        static constexpr int n = 1 << log_n;
        static constexpr int m = n/2*(n/2+1)/2;
        static constexpr int Ns = N*s;

        static void transpose_kernel(const int k, const int p, complex_vector x) noexcept
        {
            if (k != p) {
                const int p_kn = p + k*n;
                const int k_pn = k + p*n;
                const ymm aA = getpz2(x+p_kn+0);
                const ymm bB = getpz2(x+p_kn+n);
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                const ymm cC = getpz2(x+k_pn+0);
                const ymm dD = getpz2(x+k_pn+n);
                const ymm cd = catlo(cC, dD);
                const ymm CD = cathi(cC, dD);
                setpz2(x+k_pn+n, AB);
                setpz2(x+k_pn+0, ab);
                setpz2(x+p_kn+n, CD);
                setpz2(x+p_kn+0, cd);
            }
            else {
                const int k_kn = k + k*n;
                const ymm aA = getpz2(x+k_kn+0);
                const ymm bB = getpz2(x+k_kn+n);
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                setpz2(x+k_kn+n, AB);
                setpz2(x+k_kn+0, ab);
            }
        }

        static void mult_twiddle_factor_kernel(
                const int p, const int k, complex_vector x, weight_t W) noexcept
        {
            if (k != p) {
                const int kp = k*p;
                const int k_pn = k + p*n;
                const int p_kn = p + k*n;
                const xmm w11 = modqpz<Ns,s>(W,kp);
                const xmm w12 = modqpz<Ns,s>(W,kp+p);
                const ymm w1 = scalepz2<N,mode>(cat(w11, w12));
                const xmm w21 = modqpz<Ns,s>(W,kp+k);
                const xmm w22 = modqpz<Ns,s>(W,kp+k+p+1);
                const ymm w2 = scalepz2<N,mode>(cat(w21, w22));
                const ymm aA = mulpz2(w1, getpz2(x+k_pn+0));
                const ymm bB = mulpz2(w2, getpz2(x+k_pn+n));
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                const ymm cC = getpz2(x+p_kn+0);
                const ymm dD = getpz2(x+p_kn+n);
                const ymm cd = mulpz2(w1, catlo(cC, dD));
                const ymm CD = mulpz2(w2, cathi(cC, dD));
                setpz2(x+p_kn+n, AB);
                setpz2(x+p_kn+0, ab);
                setpz2(x+k_pn+n, CD);
                setpz2(x+k_pn+0, cd);
            }
            else {
                const int pp = p*p;
                const int p_pn = p + p*n;
                const xmm w11 = modqpz<Ns,s>(W,pp);
                const xmm w12 = modqpz<Ns,s>(W,pp+p);
                const xmm w22 = modqpz<Ns,s>(W,pp+2*p+1);
                const ymm w1 = scalepz2<N,mode>(cat(w11, w12));
                const ymm aA = mulpz2(w1, getpz2(x+p_pn+0));
                const ymm w2 = scalepz2<N,mode>(cat(w12, w22));
                const ymm bB = mulpz2(w2, getpz2(x+p_pn+n));
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                setpz2(x+p_pn+n, AB);
                setpz2(x+p_pn+0, ab);
            }
        }

        void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            if (N < OMP_THRESHOLD1 || sng) {
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
                for (int p = 0; p < n; p++) {
                    const int pn = p*n;
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].col;
                    const int p = ip[i].row;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
                }
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
            }
            else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
            #pragma omp parallel firstprivate(ip,x,y,W,Ws)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
                #pragma omp for schedule(static)
                for (int p = 0; p < n; p++) {
                    const int pn = p*n;
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].col;
                    const int p = ip[i].row;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(static)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
                }
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
            }
            else //////////////////////////////////////////////////////////////////
            #pragma omp parallel firstprivate(ip,x,y,W,Ws)
            {
                #pragma omp for schedule(guided)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
                #pragma omp for schedule(guided)
                for (int p = 0; p < n; p++) {
                    const int pn = p*n;
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(guided)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].col;
                    const int p = ip[i].row;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(guided)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_FourStep::fwdfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
                }
                #pragma omp for schedule(guided) nowait
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
            }
        }
    };

    template <int log_N, int mode, bool sng = 0> struct fwdffts
    {
        inline void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            fwdffts_body<log_N,1,mode,sng>()(ip, x, y, W, Ws);
        }
    };

    template <int log_N, int mode, bool sng = 0> struct fwdffts2
    {
        inline void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            fwdffts_body<log_N,2,mode,sng>()(ip, x, y, W, Ws);
        }
    };

    template <int log_N, int mode, bool sng = 0> struct fwdffts8
    {
        inline void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            fwdffts_body<log_N,8,mode,sng>()(ip, x, y, W, Ws);
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    template <int log_N, int s, int mode, bool sng> struct invffts_body
    {
        static constexpr int log_n = log_N/2;
        static constexpr int N = 1 << log_N;
        static constexpr int n = 1 << log_n;
        static constexpr int m = n/2*(n/2+1)/2;
        static constexpr int Ns = N*s;

        static inline void transpose_kernel(
                const int k, const int p, complex_vector x) noexcept
        {
            fwdffts_body<log_N,s,mode,sng>::transpose_kernel(k, p, x);
        }

        static void mult_twiddle_factor_kernel(
                const int p, const int k, complex_vector x, weight_t W) noexcept
        {
            if (k != p) {
                const int kp = k*p;
                const int k_pn = k + p*n;
                const int p_kn = p + k*n;
                const xmm w11 = modqpz<Ns,s>(W,kp);
                const xmm w12 = modqpz<Ns,s>(W,kp+p);
                const ymm w1 = scalepz2<N,mode>(cnjpz2(cat(w11, w12)));
                const xmm w21 = modqpz<Ns,s>(W,kp+k);
                const xmm w22 = modqpz<Ns,s>(W,kp+k+p+1);
                const ymm w2 = scalepz2<N,mode>(cnjpz2(cat(w21, w22)));
                const ymm aA = mulpz2(w1, getpz2(x+k_pn+0));
                const ymm bB = mulpz2(w2, getpz2(x+k_pn+n));
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                const ymm cC = getpz2(x+p_kn+0);
                const ymm dD = getpz2(x+p_kn+n);
                const ymm cd = mulpz2(w1, catlo(cC, dD));
                const ymm CD = mulpz2(w2, cathi(cC, dD));
                setpz2(x+p_kn+n, AB);
                setpz2(x+p_kn+0, ab);
                setpz2(x+k_pn+n, CD);
                setpz2(x+k_pn+0, cd);
            }
            else {
                const int pp = p*p;
                const int p_pn = p + p*n;
                const xmm w11 = modqpz<Ns,s>(W,pp);
                const xmm w12 = modqpz<Ns,s>(W,pp+p);
                const xmm w22 = modqpz<Ns,s>(W,pp+2*p+1);
                const ymm w1 = scalepz2<N,mode>(cnjpz2(cat(w11, w12)));
                const ymm aA = mulpz2(w1, getpz2(x+p_pn+0));
                const ymm w2 = scalepz2<N,mode>(cnjpz2(cat(w12, w22)));
                const ymm bB = mulpz2(w2, getpz2(x+p_pn+n));
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                setpz2(x+p_pn+n, AB);
                setpz2(x+p_pn+0, ab);
            }
        }

        void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            if (N < OMP_THRESHOLD1 || sng) {
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
                for (int p = 0; p < n; p++) {
                    const int pn = p*n;
                    OTFFT_FourStep::invfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].col;
                    const int p = ip[i].row;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_FourStep::invfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
                }
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
            }
            else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
            #pragma omp parallel firstprivate(ip,x,y,W,Ws)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
                #pragma omp for schedule(static)
                for (int p = 0; p < n; p++) {
                    const int pn = p*n;
                    OTFFT_FourStep::invfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].col;
                    const int p = ip[i].row;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(static)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_FourStep::invfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
                }
                #pragma omp for schedule(static) nowait
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
            }
            else //////////////////////////////////////////////////////////////////
            #pragma omp parallel firstprivate(ip,x,y,W,Ws)
            {
                #pragma omp for schedule(guided)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
                #pragma omp for schedule(guided)
                for (int p = 0; p < n; p++) {
                    const int pn = p*n;
                    OTFFT_FourStep::invfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(guided)
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].col;
                    const int p = ip[i].row;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(guided)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_FourStep::invfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
                }
                #pragma omp for schedule(guided) nowait
                for (int i = 0; i < m; i++) {
                    const int k = ip[i].row;
                    const int p = ip[i].col;
                    transpose_kernel(k, p, x);
                }
            }
        }
    };

    template <int log_N, int mode, bool sng = 0> struct invffts
    {
        inline void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            invffts_body<log_N,1,mode,sng>()(ip, x, y, W, Ws);
        }
    };

    template <int log_N, int mode, bool sng = 0> struct invffts2
    {
        inline void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            invffts_body<log_N,2,mode,sng>()(ip, x, y, W, Ws);
        }
    };

    template <int log_N, int mode, bool sng = 0> struct invffts8
    {
        inline void operator()(const_index_vector ip,
                complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
        {
            invffts_body<log_N,8,mode,sng>()(ip, x, y, W, Ws);
        }
    };

} /////////////////////////////////////////////////////////////////////////////

}

#endif // otfft_sixstepsq_h
