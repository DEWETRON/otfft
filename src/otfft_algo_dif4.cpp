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

#include "otfft_avxdif4.h"

#include <cassert>
#include <stdint.h>
#include <iostream>

namespace OTFFT_NAMESPACE
{
    namespace OTFFT_AVXDIF4
    {
        struct FFT0 : FFT_IF
        {
            int N, log_N;
            simd_array<complex_t> weight;
            complex_t* __restrict W;

            FFT0() noexcept : N(0), log_N(0), W(0) {}
            FFT0(const int n) { setup(n); }

            void setup(int n)
            {
                for (log_N = 0; n > 1; n >>= 1) log_N++;
                setup2(log_N);
            }

            void setup2(const int n)
            {
                log_N = n; N = 1 << n;
                weight.setup(N+1); W = &weight;
                init_W(N, W);
            }

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

        void FFT0::fwd(complex_vector x, complex_vector y) const noexcept
        {
            static const int mode = scale_length;
            if (N < OMP_THRESHOLD) {
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
            else OTFFT_AVXDIF4omp::fwd(log_N, x, y, W);
        }

        void FFT0::fwd0(complex_vector x, complex_vector y) const noexcept
        {
            static const int mode = scale_1;
            if (N < OMP_THRESHOLD) {
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
            else OTFFT_AVXDIF4omp::fwd0(log_N, x, y, W);
        }

        void FFT0::fwdn(complex_vector x, complex_vector y) const noexcept
        {
            fwd(x, y);
        }

        void FFT0::fwdu(complex_vector x, complex_vector y) const noexcept
        {
            static const int mode = scale_unitary;
            if (N < OMP_THRESHOLD) {
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
            else OTFFT_AVXDIF4omp::fwdu(log_N, x, y, W);
        }

        void FFT0::inv(complex_vector x, complex_vector y) const noexcept
        {
            static const int mode = scale_1;
            if (N < OMP_THRESHOLD) {
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
            else OTFFT_AVXDIF4omp::inv(log_N, x, y, W);
        }

        void FFT0::inv0(complex_vector x, complex_vector y) const noexcept
        {
            inv(x, y);
        }

        void FFT0::invn(complex_vector x, complex_vector y) const noexcept
        {
            static const int mode = scale_length;
            if (N < OMP_THRESHOLD) {
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
            else OTFFT_AVXDIF4omp::invn(log_N, x, y, W);
        }

        void FFT0::invu(complex_vector x, complex_vector y) const noexcept
        {
            static const int mode = scale_unitary;
            if (N < OMP_THRESHOLD) {
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
            else OTFFT_AVXDIF4omp::invu(log_N, x, y, W);
        }
    }
}
