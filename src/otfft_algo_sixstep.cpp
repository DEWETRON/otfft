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

#include "otfft_sixstep.h"

#include <cassert>
#include <stdint.h>
#include <iostream>

namespace OTFFT_NAMESPACE
{
    namespace OTFFT_SixStep
    {
        using namespace OTFFT_EightStep;

        struct FFT0 : FFT_IF
        {
            typedef complex_t* __restrict complex_ptr;
            typedef index_t* __restrict index_ptr;
            int N, log_N;
            simd_array<complex_t> weight;
            simd_array<complex_t> weight_sub;
            simd_array<index_t> index;
            complex_ptr W;
            complex_ptr Ws;
            index_ptr iv;

            FFT0() noexcept;
            FFT0(const int n);

            void setup(int n);

            void setup2(const int n);

            ///////////////////////////////////////////////////////////////////////////

            void fwd(complex_vector x, complex_vector y) const noexcept;
            void fwd0(complex_vector x, complex_vector y) const noexcept;
            void fwdu(complex_vector x, complex_vector y) const noexcept;
            void fwdn(complex_vector x, complex_vector y) const noexcept;

            ///////////////////////////////////////////////////////////////////////////

            void inv(complex_vector x, complex_vector y) const noexcept;
            void inv0(complex_vector x, complex_vector y) const noexcept;
            void invu(complex_vector x, complex_vector y) const noexcept;
            void invn(complex_vector x, complex_vector y) const noexcept;
        };

        std::unique_ptr<FFT_IF> instance() { return std::unique_ptr<FFT_IF>(new FFT0()); }

        FFT0::FFT0() noexcept : N(0), log_N(0), W(0), Ws(0), iv(0) {}

        FFT0::FFT0(const int n)
        {
            setup(n);
        }

        void FFT0::setup(int n)
        {
            for (log_N = 0; n > 1; n >>= 1) log_N++;
            setup2(log_N);
        }

        void FFT0::setup2(const int n)
        {
            log_N = n; N = 1 << n;
            weight.setup(N+1); W = &weight; init_W(N, W);
            if (n < 4) {}
            else if ((n & 1) == 1) {
                const int m = 1 << (n/2-1);
                weight_sub.setup(m+1); Ws = &weight_sub; init_W(m, Ws);
                index.setup(m/2*(m/2+1)/2); iv = &index;
                int i = 0;
                for (int k = 0; k < m; k += 2) {
                    for (int p = k; p < m; p += 2) {
                        iv[i].row = k;
                        iv[i].col = p;
                        i++;
                    }
                }
            }
            else {
                const int m = 1 << n/2;
                weight_sub.setup(m+1); Ws = &weight_sub; init_W(m, Ws);
                index.setup(m/2*(m/2+1)/2); iv = &index;
                int i = 0;
                for (int k = 0; k < m; k += 2) {
                    for (int p = k; p < m; p += 2) {
                        iv[i].row = k;
                        iv[i].col = p;
                        i++;
                    }
                }
            }
        }

        inline void FFT0::fwd(complex_vector x, complex_vector y) const noexcept
        {
            constexpr int mode = scale_length;
            switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::fwdfft<(1<<1),1,0,mode>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::fwdfft<(1<<2),1,0,mode>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::fwdfft<(1<<3),1,0,mode>()(x, y, W); break;
            case  4: fwdffts< 4,mode>()(iv, x, y, W, Ws); break;
            case  5: fwdfftr< 5,mode>()(iv, x, y, W, Ws); break;
            case  6: fwdffts< 6,mode>()(iv, x, y, W, Ws); break;
            case  7: fwdfftr< 7,mode>()(iv, x, y, W, Ws); break;
            case  8: fwdffts< 8,mode>()(iv, x, y, W, Ws); break;
            case  9: fwdfftr< 9,mode>()(iv, x, y, W, Ws); break;
            case 10: fwdffts<10,mode>()(iv, x, y, W, Ws); break;
            case 11: fwdfftr<11,mode>()(iv, x, y, W, Ws); break;
            case 12: fwdffts<12,mode>()(iv, x, y, W, Ws); break;
            case 13: fwdfftr<13,mode>()(iv, x, y, W, Ws); break;
            case 14: fwdffts<14,mode>()(iv, x, y, W, Ws); break;
            case 15: fwdfftr<15,mode>()(iv, x, y, W, Ws); break;
            case 16: fwdffts<16,mode>()(iv, x, y, W, Ws); break;
            case 17: fwdfftr<17,mode>()(iv, x, y, W, Ws); break;
            case 18: fwdffts<18,mode>()(iv, x, y, W, Ws); break;
            case 19: fwdfftr<19,mode>()(iv, x, y, W, Ws); break;
            case 20: fwdffts<20,mode>()(iv, x, y, W, Ws); break;
            case 21: fwdfftr<21,mode>()(iv, x, y, W, Ws); break;
            case 22: fwdffts<22,mode>()(iv, x, y, W, Ws); break;
            case 23: fwdfftr<23,mode>()(iv, x, y, W, Ws); break;
            case 24: fwdffts<24,mode>()(iv, x, y, W, Ws); break;
            }
        }

        inline void FFT0::fwd0(complex_vector x, complex_vector y) const noexcept
        {
            constexpr int mode = scale_1;
            switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::fwdfft<(1<<1),1,0,mode>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::fwdfft<(1<<2),1,0,mode>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::fwdfft<(1<<3),1,0,mode>()(x, y, W); break;
            case  4: fwdffts< 4,mode>()(iv, x, y, W, Ws); break;
            case  5: fwdfftr< 5,mode>()(iv, x, y, W, Ws); break;
            case  6: fwdffts< 6,mode>()(iv, x, y, W, Ws); break;
            case  7: fwdfftr< 7,mode>()(iv, x, y, W, Ws); break;
            case  8: fwdffts< 8,mode>()(iv, x, y, W, Ws); break;
            case  9: fwdfftr< 9,mode>()(iv, x, y, W, Ws); break;
            case 10: fwdffts<10,mode>()(iv, x, y, W, Ws); break;
            case 11: fwdfftr<11,mode>()(iv, x, y, W, Ws); break;
            case 12: fwdffts<12,mode>()(iv, x, y, W, Ws); break;
            case 13: fwdfftr<13,mode>()(iv, x, y, W, Ws); break;
            case 14: fwdffts<14,mode>()(iv, x, y, W, Ws); break;
            case 15: fwdfftr<15,mode>()(iv, x, y, W, Ws); break;
            case 16: fwdffts<16,mode>()(iv, x, y, W, Ws); break;
            case 17: fwdfftr<17,mode>()(iv, x, y, W, Ws); break;
            case 18: fwdffts<18,mode>()(iv, x, y, W, Ws); break;
            case 19: fwdfftr<19,mode>()(iv, x, y, W, Ws); break;
            case 20: fwdffts<20,mode>()(iv, x, y, W, Ws); break;
            case 21: fwdfftr<21,mode>()(iv, x, y, W, Ws); break;
            case 22: fwdffts<22,mode>()(iv, x, y, W, Ws); break;
            case 23: fwdfftr<23,mode>()(iv, x, y, W, Ws); break;
            case 24: fwdffts<24,mode>()(iv, x, y, W, Ws); break;
            }
        }

        void FFT0::fwdu(complex_vector x, complex_vector y) const noexcept
        {
            constexpr int mode = scale_unitary;
            switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::fwdfft<(1<<1),1,0,mode>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::fwdfft<(1<<2),1,0,mode>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::fwdfft<(1<<3),1,0,mode>()(x, y, W); break;
            case  4: fwdffts< 4,mode>()(iv, x, y, W, Ws); break;
            case  5: fwdfftr< 5,mode>()(iv, x, y, W, Ws); break;
            case  6: fwdffts< 6,mode>()(iv, x, y, W, Ws); break;
            case  7: fwdfftr< 7,mode>()(iv, x, y, W, Ws); break;
            case  8: fwdffts< 8,mode>()(iv, x, y, W, Ws); break;
            case  9: fwdfftr< 9,mode>()(iv, x, y, W, Ws); break;
            case 10: fwdffts<10,mode>()(iv, x, y, W, Ws); break;
            case 11: fwdfftr<11,mode>()(iv, x, y, W, Ws); break;
            case 12: fwdffts<12,mode>()(iv, x, y, W, Ws); break;
            case 13: fwdfftr<13,mode>()(iv, x, y, W, Ws); break;
            case 14: fwdffts<14,mode>()(iv, x, y, W, Ws); break;
            case 15: fwdfftr<15,mode>()(iv, x, y, W, Ws); break;
            case 16: fwdffts<16,mode>()(iv, x, y, W, Ws); break;
            case 17: fwdfftr<17,mode>()(iv, x, y, W, Ws); break;
            case 18: fwdffts<18,mode>()(iv, x, y, W, Ws); break;
            case 19: fwdfftr<19,mode>()(iv, x, y, W, Ws); break;
            case 20: fwdffts<20,mode>()(iv, x, y, W, Ws); break;
            case 21: fwdfftr<21,mode>()(iv, x, y, W, Ws); break;
            case 22: fwdffts<22,mode>()(iv, x, y, W, Ws); break;
            case 23: fwdfftr<23,mode>()(iv, x, y, W, Ws); break;
            case 24: fwdffts<24,mode>()(iv, x, y, W, Ws); break;
            }
        }

        inline void FFT0::fwdn(complex_vector x, complex_vector y) const noexcept { fwd(x, y); }

        inline void FFT0::inv(complex_vector x, complex_vector y) const noexcept
        {
            constexpr int mode = scale_1;
            switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::invfft<(1<<1),1,0,mode>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::invfft<(1<<2),1,0,mode>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::invfft<(1<<3),1,0,mode>()(x, y, W); break;
            case  4: invffts< 4,mode>()(iv, x, y, W, Ws); break;
            case  5: invfftr< 5,mode>()(iv, x, y, W, Ws); break;
            case  6: invffts< 6,mode>()(iv, x, y, W, Ws); break;
            case  7: invfftr< 7,mode>()(iv, x, y, W, Ws); break;
            case  8: invffts< 8,mode>()(iv, x, y, W, Ws); break;
            case  9: invfftr< 9,mode>()(iv, x, y, W, Ws); break;
            case 10: invffts<10,mode>()(iv, x, y, W, Ws); break;
            case 11: invfftr<11,mode>()(iv, x, y, W, Ws); break;
            case 12: invffts<12,mode>()(iv, x, y, W, Ws); break;
            case 13: invfftr<13,mode>()(iv, x, y, W, Ws); break;
            case 14: invffts<14,mode>()(iv, x, y, W, Ws); break;
            case 15: invfftr<15,mode>()(iv, x, y, W, Ws); break;
            case 16: invffts<16,mode>()(iv, x, y, W, Ws); break;
            case 17: invfftr<17,mode>()(iv, x, y, W, Ws); break;
            case 18: invffts<18,mode>()(iv, x, y, W, Ws); break;
            case 19: invfftr<19,mode>()(iv, x, y, W, Ws); break;
            case 20: invffts<20,mode>()(iv, x, y, W, Ws); break;
            case 21: invfftr<21,mode>()(iv, x, y, W, Ws); break;
            case 22: invffts<22,mode>()(iv, x, y, W, Ws); break;
            case 23: invfftr<23,mode>()(iv, x, y, W, Ws); break;
            case 24: invffts<24,mode>()(iv, x, y, W, Ws); break;
            }
        }

        inline void FFT0::inv0(complex_vector x, complex_vector y) const noexcept { inv(x, y); }

        void FFT0::invn(complex_vector x, complex_vector y) const noexcept
        {
            constexpr int mode = scale_length;
            switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::invfft<(1<<1),1,0,mode>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::invfft<(1<<2),1,0,mode>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::invfft<(1<<3),1,0,mode>()(x, y, W); break;
            case  4: invffts< 4,mode>()(iv, x, y, W, Ws); break;
            case  5: invfftr< 5,mode>()(iv, x, y, W, Ws); break;
            case  6: invffts< 6,mode>()(iv, x, y, W, Ws); break;
            case  7: invfftr< 7,mode>()(iv, x, y, W, Ws); break;
            case  8: invffts< 8,mode>()(iv, x, y, W, Ws); break;
            case  9: invfftr< 9,mode>()(iv, x, y, W, Ws); break;
            case 10: invffts<10,mode>()(iv, x, y, W, Ws); break;
            case 11: invfftr<11,mode>()(iv, x, y, W, Ws); break;
            case 12: invffts<12,mode>()(iv, x, y, W, Ws); break;
            case 13: invfftr<13,mode>()(iv, x, y, W, Ws); break;
            case 14: invffts<14,mode>()(iv, x, y, W, Ws); break;
            case 15: invfftr<15,mode>()(iv, x, y, W, Ws); break;
            case 16: invffts<16,mode>()(iv, x, y, W, Ws); break;
            case 17: invfftr<17,mode>()(iv, x, y, W, Ws); break;
            case 18: invffts<18,mode>()(iv, x, y, W, Ws); break;
            case 19: invfftr<19,mode>()(iv, x, y, W, Ws); break;
            case 20: invffts<20,mode>()(iv, x, y, W, Ws); break;
            case 21: invfftr<21,mode>()(iv, x, y, W, Ws); break;
            case 22: invffts<22,mode>()(iv, x, y, W, Ws); break;
            case 23: invfftr<23,mode>()(iv, x, y, W, Ws); break;
            case 24: invffts<24,mode>()(iv, x, y, W, Ws); break;
            }
        }

        void FFT0::invu(complex_vector x, complex_vector y) const noexcept
        {
            constexpr int mode = scale_unitary;
            switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::invfft<(1<<1),1,0,mode>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::invfft<(1<<2),1,0,mode>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::invfft<(1<<3),1,0,mode>()(x, y, W); break;
            case  4: invffts< 4,mode>()(iv, x, y, W, Ws); break;
            case  5: invfftr< 5,mode>()(iv, x, y, W, Ws); break;
            case  6: invffts< 6,mode>()(iv, x, y, W, Ws); break;
            case  7: invfftr< 7,mode>()(iv, x, y, W, Ws); break;
            case  8: invffts< 8,mode>()(iv, x, y, W, Ws); break;
            case  9: invfftr< 9,mode>()(iv, x, y, W, Ws); break;
            case 10: invffts<10,mode>()(iv, x, y, W, Ws); break;
            case 11: invfftr<11,mode>()(iv, x, y, W, Ws); break;
            case 12: invffts<12,mode>()(iv, x, y, W, Ws); break;
            case 13: invfftr<13,mode>()(iv, x, y, W, Ws); break;
            case 14: invffts<14,mode>()(iv, x, y, W, Ws); break;
            case 15: invfftr<15,mode>()(iv, x, y, W, Ws); break;
            case 16: invffts<16,mode>()(iv, x, y, W, Ws); break;
            case 17: invfftr<17,mode>()(iv, x, y, W, Ws); break;
            case 18: invffts<18,mode>()(iv, x, y, W, Ws); break;
            case 19: invfftr<19,mode>()(iv, x, y, W, Ws); break;
            case 20: invffts<20,mode>()(iv, x, y, W, Ws); break;
            case 21: invfftr<21,mode>()(iv, x, y, W, Ws); break;
            case 22: invffts<22,mode>()(iv, x, y, W, Ws); break;
            case 23: invfftr<23,mode>()(iv, x, y, W, Ws); break;
            case 24: invffts<24,mode>()(iv, x, y, W, Ws); break;
            }
        }
    }
}
