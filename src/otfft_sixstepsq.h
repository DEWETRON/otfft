// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

#include "otfft_types.h"
#include "otfft_avxdif16.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_Sixstep { /////////////////////////////////////////////////////

    static const int OMP_THRESHOLD1 = 1<<13;
    static const int OMP_THRESHOLD2 = 1<<17;

    template <int log_N, int s, int mode, bool sng> struct fwdffts_body
    {
        static constexpr int log_n = log_N/2;
        static constexpr int N = 1 << log_N;
        static constexpr int n = 1 << log_n;
        static constexpr int m = n/2*(n/2+1)/2;
        static constexpr int Ns = N*s;

        static void transpose_kernel(const int k, const int p, complex_vector x) noexcept
        {
            if (k == p) {
                const int k_kn = k + k*n;
                const ymm aA = getpz2(x+k_kn+0);
                const ymm bB = getpz2(x+k_kn+n);
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                setpz2(x+k_kn+0, ab);
                setpz2(x+k_kn+n, AB);
            }
            else {
                const int p_kn = p + k*n;
                const int k_pn = k + p*n;
                const ymm aA = getpz2(x+p_kn+0);
                const ymm bB = getpz2(x+p_kn+n);
                const ymm cC = getpz2(x+k_pn+0);
                const ymm dD = getpz2(x+k_pn+n);
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                const ymm cd = catlo(cC, dD);
                const ymm CD = cathi(cC, dD);
                setpz2(x+k_pn+0, ab);
                setpz2(x+k_pn+n, AB);
                setpz2(x+p_kn+0, cd);
                setpz2(x+p_kn+n, CD);
            }
        }

        static void mult_twiddle_factor_kernel(
                const int p, const int k, complex_vector x, weight_t W) noexcept
        {
            if (p == k) {
                const int pp = p*p;
                complex_vector x_p_pn = x + p + p*n;
                const complex_t& w = W[s*(pp+p)];
                const ymm w1 = scalepz2<N,mode>(cmplx2(W[s*(pp)], w));
                const ymm w2 = scalepz2<N,mode>(cmplx2(w, W[s*(pp+2*p+1)]));
                const ymm aA = mulpz2(w1, getpz2(x_p_pn+0));
                const ymm bB = mulpz2(w2, getpz2(x_p_pn+n));
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                setpz2(x_p_pn+0, ab);
                setpz2(x_p_pn+n, AB);
            }
            else {
                const int kp = k*p;
                complex_vector x_k_pn = x + k + p*n;
                complex_vector x_p_kn = x + p + k*n;
                const ymm w1 = scalepz2<N,mode>(cmplx2(W[s*(kp)],   W[s*(kp+p)]));
                const ymm w2 = scalepz2<N,mode>(cmplx2(W[s*(kp+k)], W[s*(kp+k+p+1)]));
                const ymm aA = mulpz2(w1, getpz2(x_k_pn+0));
                const ymm bB = mulpz2(w2, getpz2(x_k_pn+n));
                const ymm cC = getpz2(x_p_kn+0);
                const ymm dD = getpz2(x_p_kn+n);
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                const ymm cd = mulpz2(w1, catlo(cC, dD));
                const ymm CD = mulpz2(w2, cathi(cC, dD));
                setpz2(x_p_kn+0, ab);
                setpz2(x_p_kn+n, AB);
                setpz2(x_k_pn+0, cd);
                setpz2(x_k_pn+n, CD);
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
                    OTFFT_AVXDIF16::fwdfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                for (int i = 0; i < m; i++) {
                    const int p = ip[i].row;
                    const int k = ip[i].col;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_AVXDIF16::fwdfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
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
                    OTFFT_AVXDIF16::fwdfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < m; i++) {
                    const int p = ip[i].row;
                    const int k = ip[i].col;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(static)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_AVXDIF16::fwdfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
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
                    OTFFT_AVXDIF16::fwdfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(guided)
                for (int i = 0; i < m; i++) {
                    const int p = ip[i].row;
                    const int k = ip[i].col;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(guided)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_AVXDIF16::fwdfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
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
            if (p == k) {
                const int M = N-p*p;
                complex_vector x_p_pn = x + p + p*n;
                const complex_t& w = W[s*(M-p)];
                const ymm w1 = scalepz2<N,mode>(cmplx2(W[s*(M)], w));
                const ymm w2 = scalepz2<N,mode>(cmplx2(w, W[s*(M-2*p-1)]));
                const ymm aA = mulpz2(w1, getpz2(x_p_pn+0));
                const ymm bB = mulpz2(w2, getpz2(x_p_pn+n));
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                setpz2(x_p_pn+0, ab);
                setpz2(x_p_pn+n, AB);
            }
            else {
                const int M = N-k*p;
                complex_vector x_k_pn = x + k + p*n;
                complex_vector x_p_kn = x + p + k*n;
                const ymm w1 = scalepz2<N,mode>(cmplx2(W[s*(M)],   W[s*(M-p)]));
                const ymm w2 = scalepz2<N,mode>(cmplx2(W[s*(M-k)], W[s*(M-k-p-1)]));
                const ymm aA = mulpz2(w1, getpz2(x_k_pn+0));
                const ymm bB = mulpz2(w2, getpz2(x_k_pn+n));
                const ymm cC = getpz2(x_p_kn+0);
                const ymm dD = getpz2(x_p_kn+n);
                const ymm ab = catlo(aA, bB);
                const ymm AB = cathi(aA, bB);
                const ymm cd = mulpz2(w1, catlo(cC, dD));
                const ymm CD = mulpz2(w2, cathi(cC, dD));
                setpz2(x_p_kn+0, ab);
                setpz2(x_p_kn+n, AB);
                setpz2(x_k_pn+0, cd);
                setpz2(x_k_pn+n, CD);
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
                    OTFFT_AVXDIF16::invfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                for (int i = 0; i < m; i++) {
                    const int p = ip[i].row;
                    const int k = ip[i].col;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_AVXDIF16::invfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
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
                    OTFFT_AVXDIF16::invfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(static)
                for (int i = 0; i < m; i++) {
                    const int p = ip[i].row;
                    const int k = ip[i].col;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(static)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_AVXDIF16::invfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
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
                    OTFFT_AVXDIF16::invfft<n,1,0,scale_1>()(x + pn, y + pn, Ws);
                }
                #pragma omp for schedule(guided)
                for (int i = 0; i < m; i++) {
                    const int p = ip[i].row;
                    const int k = ip[i].col;
                    mult_twiddle_factor_kernel(p, k, x, W);
                }
                #pragma omp for schedule(guided)
                for (int k = 0; k < n; k++) {
                    const int kn = k*n;
                    OTFFT_AVXDIF16::invfft<n,1,0,scale_1>()(x + kn, y + kn, Ws);
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
