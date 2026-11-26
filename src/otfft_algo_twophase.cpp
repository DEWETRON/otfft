// Copyright DEWETRON GmbH 2026
// TwoPhase: see otfft_twophase.h and decision D35. Based on the kernels of OK-Ojisan Template FFT.

#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"

#ifdef USE_AVX2
#include "otfft_avx2.h"
#endif
#ifdef USE_AVX512
#include "otfft_avx512.h"
#endif
#ifdef USE_AVX
#include "otfft_avx.h"
#endif
#ifdef USE_SSE2
#include "otfft_sse2.h"
#endif

#include "otfft_twophase.h"

#include <cmath>
#include <stdexcept>

namespace OTFFT_NAMESPACE
{
    namespace OTFFT_TwoPhase
    {
        struct FFT0 : FFT_IF
        {
            int N, log_N;
            simd_array<complex_t> weight1, weight2;
            simd_array<complex_t> tile_array;
            std::vector<complex_t> lo;
            std::vector<complex_t> hi[3];   // for the scale factors 1, 1/N and 1/sqrt(N)
            Context cx;

            FFT0() noexcept : N(0), log_N(0), cx() {}
            FFT0(const int n) { setup(n); }

            void setup(int n)
            {
                int l = 0;
                for (; n > 1; n >>= 1) l++;
                setup2(l);
            }

            void setup2(const int n)
            {
                if (n < MIN_LOG || n > MAX_LOG)
                    throw std::domain_error("OTFFT TwoPhase supports the lengths 2^16 to 2^22 only");
                log_N = n; N = 1 << n;
                int logm = 0, C = 0;
                shape(n, logm, C);
                const int M = 1 << logm, R = N / M;

                weight1.setup(2*M*C);
                weight2.setup(2*R*C);
                cx.W1 = &weight1;
                cx.W2 = &weight2;
                init_Wt(8, M*C, cx.W1);
                init_Wt(8, R*C, cx.W2);

                lo.resize(M);
                for (int i = 0; i < M; i++) lo[i] = unit(i, N);
                const double scales[3] = {1.0, 1.0/N, 1.0/std::sqrt(double(N))};
                for (int t = 0; t < 3; t++) {
                    hi[t].resize(R);
                    for (int j = 0; j < R; j++) {
                        const complex_t w = unit((long long)j*M, N);
                        hi[t][j] = complex_t(w.Re*scales[t], w.Im*scales[t]);
                    }
                }

                int nt = 1;
#ifdef _OPENMP
                nt = omp_get_max_threads();
#endif
                cx.ntiles = nt;
                cx.tile_size = size_t(M*C > R*C ? M*C : R*C);
                tile_array.setup(2*size_t(nt)*cx.tile_size);
                cx.tiles = &tile_array;
                cx.lo = lo.data();
            }

            static void shape(const int n, int& logm, int& c)
            {
                switch (n) {
                case 16: logm = Shape<16>::logm; c = Shape<16>::c; break;
                case 17: logm = Shape<17>::logm; c = Shape<17>::c; break;
                case 18: logm = Shape<18>::logm; c = Shape<18>::c; break;
                case 19: logm = Shape<19>::logm; c = Shape<19>::c; break;
                case 20: logm = Shape<20>::logm; c = Shape<20>::c; break;
                case 21: logm = Shape<21>::logm; c = Shape<21>::c; break;
                default: logm = Shape<22>::logm; c = Shape<22>::c; break;
                }
            }

            static complex_t unit(const long long e, const int n)
            {
                const long double theta = -2.0L * 3.14159265358979323846264338327950288L * (long double)e / (long double)n;
                return complex_t(double(cosl(theta)), double(sinl(theta)));
            }

            template <bool inv> void transform(const int scale, complex_vector x, complex_vector y) const noexcept
            {
                Context c = cx;
                c.hi = hi[scale].data();
                switch (log_N) {
                case 16: Pass<16,inv>::run(c, x, y); break;
                case 17: Pass<17,inv>::run(c, x, y); break;
                case 18: Pass<18,inv>::run(c, x, y); break;
                case 19: Pass<19,inv>::run(c, x, y); break;
                case 20: Pass<20,inv>::run(c, x, y); break;
                case 21: Pass<21,inv>::run(c, x, y); break;
                case 22: Pass<22,inv>::run(c, x, y); break;
                }
            }

            // scale: 0 = none, 1 = 1/N, 2 = 1/sqrt(N)
            void fwd(complex_vector x, complex_vector y) const noexcept { transform<false>(1, x, y); }
            void fwd0(complex_vector x, complex_vector y) const noexcept { transform<false>(0, x, y); }
            void fwdu(complex_vector x, complex_vector y) const noexcept { transform<false>(2, x, y); }
            void fwdn(complex_vector x, complex_vector y) const noexcept { transform<false>(1, x, y); }

            void inv(complex_vector x, complex_vector y) const noexcept { transform<true>(0, x, y); }
            void inv0(complex_vector x, complex_vector y) const noexcept { transform<true>(0, x, y); }
            void invu(complex_vector x, complex_vector y) const noexcept { transform<true>(2, x, y); }
            void invn(complex_vector x, complex_vector y) const noexcept { transform<true>(1, x, y); }
        };

        std::unique_ptr<FFT_IF> instance() { return std::unique_ptr<FFT_IF>(new FFT0()); }
    }
}
