// Copyright DEWETRON GmbH 2017
// based on OK-Ojisan Template FFT

#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"

#ifdef USE_AVX2
#include "otfft_avx2.h"
#endif
#ifdef USE_AVX
#include "otfft_avx.h"
#endif
#ifdef USE_SSE2
#include "otfft_sse2.h"
#endif

#include "otfft_mixedradix.h"

#include <cassert>
#include <stdint.h>
#include <iostream>

namespace OTFFT_NAMESPACE
{
    namespace OTFFT_MixedRadix
    {
        struct FFT0 : FFT_IF
        {
            int N;
            simd_array<complex_t> weight;
            complex_t* __restrict W;

            FFT0() noexcept : N(0), W(0) {}
            FFT0(const int n) { setup(n); }

            void setup(const int n)
            {
                const double theta0 = 2*CONSTANT::PI/n;
                N = n;
                weight.setup(n+1); W = &weight;
                if (n < OMP_THRESHOLD) {
                    for (int p = 0; p <= n; p++) {
                        W[p] = complex_t(cos(p*theta0), -sin(p*theta0));
                    }
                }
                else {
    #pragma omp parallel for
                    for (int p = 0; p <= n; p++) {
                        W[p] = complex_t(cos(p*theta0), -sin(p*theta0));
                    }
                }
            }

            inline void setup2(const int n) { setup(1 << n); }

            ///////////////////////////////////////////////////////////////////////////

            void fwd(complex_vector x, complex_vector y) const noexcept
            {
                const xmm rN = cmplx(1.0/N, 1.0/N);
                if (N < OMP_THRESHOLD) {
                    fwdfft(N, 1, 0, x, y, W);
                    for (int k = 0; k < N; k++) setpz(x[k], mulpd(rN, getpz(x[k])));
                }
                else {
    #pragma omp parallel
                    fwdfftp(N, 1, 0, x, y, W);
    #pragma omp parallel for
                    for (int k = 0; k < N; k++) setpz(x[k], mulpd(rN, getpz(x[k])));
                }
            }

            void fwd0(complex_vector x, complex_vector y) const noexcept
            {
                if (N < OMP_THRESHOLD) {
                    fwdfft(N, 1, 0, x, y, W);
                }
                else {
    #pragma omp parallel
                    fwdfftp(N, 1, 0, x, y, W);
                }
            }

            void fwdu(complex_vector x, complex_vector y) const noexcept
            {
                const double ssrN = sqrt(1.0/N);
                const xmm srN = cmplx(ssrN, ssrN);
                if (N < OMP_THRESHOLD) {
                    fwdfft(N, 1, 0, x, y, W);
                    for (int k = 0; k < N; k++) setpz(x[k], mulpd(srN, getpz(x[k])));
                }
                else {
    #pragma omp parallel
                    fwdfftp(N, 1, 0, x, y, W);
    #pragma omp parallel for
                    for (int k = 0; k < N; k++) setpz(x[k], mulpd(srN, getpz(x[k])));
                }
            }

            inline void fwdn(complex_vector x, complex_vector y) const noexcept
            {
                fwd(x, y);
            }

            ///////////////////////////////////////////////////////////////////////////

            void inv(complex_vector x, complex_vector y) const noexcept
            {
                if (N < OMP_THRESHOLD) {
                    invfft(N, 1, 0, x, y, W);
                }
                else {
    #pragma omp parallel
                    invfftp(N, 1, 0, x, y, W);
                }
            }

            inline void inv0(complex_vector x, complex_vector y) const noexcept
            {
                inv(x, y);
            }

            void invu(complex_vector x, complex_vector y) const noexcept
            {
                const double ssrN = sqrt(1.0/N);
                const xmm srN = cmplx(ssrN, ssrN);
                if (N < OMP_THRESHOLD) {
                    invfft(N, 1, 0, x, y, W);
                    for (int p = 0; p < N; p++) setpz(x[p], mulpd(srN, getpz(x[p])));
                }
                else {
    #pragma omp parallel
                    invfftp(N, 1, 0, x, y, W);
    #pragma omp parallel for
                    for (int p = 0; p < N; p++) setpz(x[p], mulpd(srN, getpz(x[p])));
                }
            }

            void invn(complex_vector x, complex_vector y) const noexcept
            {
                const xmm rN = cmplx(1.0/N, 1.0/N);
                if (N < OMP_THRESHOLD) {
                    invfft(N, 1, 0, x, y, W);
                    for (int p = 0; p < N; p++) setpz(x[p], mulpd(rN, getpz(x[p])));
                }
                else {
    #pragma omp parallel
                    invfftp(N, 1, 0, x, y, W);
    #pragma omp parallel for
                    for (int p = 0; p < N; p++) setpz(x[p], mulpd(rN, getpz(x[p])));
                }
            }
        };

        std::unique_ptr<FFT_IF> instance() { return std::unique_ptr<FFT_IF>(new FFT0()); }
    }
}
