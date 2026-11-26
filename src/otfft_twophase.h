/******************************************************************************
*  OTFFT TwoPhase (DEWETRON fork; decision D35)
*
*  A threaded complex FFT in two parallel phases, built from the Stockham kernels of the
*  radix-8 class (otfft_avxdif8.h). The plans that FFTW prints for several threads have this
*  shape; it moves the data through memory about twice instead of once per pass.
*
*  N = R * M, x[n] with n = R*m + r (m < M, r < R). With F_r the FFT of size M of x[R*m + r]:
*      X[k + q*M] = sum over r of  W_R^(r*q) * ( W_N^(r*k) * F_r[k] )
*
*  Phase 1: the R child FFTs of size M. A work item is a strip of C children: it is gathered
*           into a tile of M*C values that stays in the L2 cache, transformed completely there
*           (the Stockham kernel with n = M and s = C does C interleaved FFTs), and stored to
*           y[k*R + r].
*  Phase 2: a work item is a strip of C columns k (C = B): the values y[k*R + r] are multiplied
*           by W_N^(r*k) while they are gathered into a tile (r*C values), the kernel with
*           n = R and s = C does the R-point FFTs of the C columns, and the result goes to
*           x[k + q*M].
*  Both phases run in one OpenMP parallel region (one barrier between them). x is input and
*  output, y is the work array of N values. The scale factor of the six transform functions is
*  folded into the table of the factors W_N^(r*k), so the kernels always run unscaled.
******************************************************************************/

#ifndef otfft_twophase_h
#define otfft_twophase_h

#include "otfft_avxdif8.h"
#include "otfft_tuning.h"

#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace OTFFT_NAMESPACE {

namespace OTFFT_TwoPhase { /////////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;
    namespace K = OTFFT_AVXDIF8;

    // Lengths 2^MIN_LOG to 2^MAX_LOG are supported; they are the ones that were measured.
    constexpr int MIN_LOG = OTFFT::Tuning::TWOPHASE_MIN_LOG;
    constexpr int MAX_LOG = OTFFT::Tuning::TWOPHASE_MAX_LOG;

    // log2 of the child size M and the strip width C for each length. Chosen with the sweep of
    // docs/benchmarks.md ("Two-phase FFT"): the tile of M*C values and the tile of R*C values
    // both stay at or below 2^13 values (128 KB), so they fit in the L2 cache with the weights.
    template <int LOGN> struct Shape;
    template <> struct Shape<16> { static constexpr int logm =  9, c = 16; };
    template <> struct Shape<17> { static constexpr int logm =  9, c =  4; };
    template <> struct Shape<18> { static constexpr int logm =  9, c =  4; };
    template <> struct Shape<19> { static constexpr int logm = 10, c =  4; };
    template <> struct Shape<20> { static constexpr int logm = 10, c =  8; };
    template <> struct Shape<21> { static constexpr int logm = 10, c =  4; };
    template <> struct Shape<22> { static constexpr int logm = 11, c =  4; };

    // everything the passes read; owned by FFT0 in otfft_algo_twophase.cpp
    struct Context
    {
        complex_t* W1;            // weights of the child FFTs: table for n*s = M*C
        complex_t* W2;            // weights of the column FFTs: table for n*s = R*C
        const complex_t* lo;      // W_N^e for e < M          (forward sign; inverse uses the conjugate)
        const complex_t* hi;      // scale * W_N^(M*j), j < R (forward sign; the inverse conjugates the product)
        complex_t* tiles;         // two tiles of tile_size values per thread
        size_t tile_size;
        int ntiles;
    };

    inline complex_t cmul(const complex_t a, const complex_t b) noexcept
    {
        return complex_t(a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re);
    }
    inline complex_t cconj(const complex_t a) noexcept { return complex_t(a.Re, -a.Im); }

    template <int n, int s, bool inv> struct Kernel;
    template <int n, int s> struct Kernel<n, s, false>
    {
        static void run(complex_vector x, complex_vector y, const_complex_vector W) noexcept
        {
            K::fwdfft<n,s,false,scale_1>()(x, y, W);
        }
    };
    template <int n, int s> struct Kernel<n, s, true>
    {
        static void run(complex_vector x, complex_vector y, const_complex_vector W) noexcept
        {
            K::invfft<n,s,false,scale_1>()(x, y, W);
        }
    };

    template <int LOGN, bool inv> struct Pass
    {
        static constexpr int logm = Shape<LOGN>::logm;
        static constexpr int C = Shape<LOGN>::c;
        static constexpr int M = 1 << logm;
        static constexpr int N = 1 << LOGN;
        static constexpr int R = N / M;

        static void run(const Context& cx, complex_vector x, complex_vector y) noexcept
        {
#ifdef _OPENMP
            // never more threads than tiles (set at plan time) or than the user allows now
            const int nthreads = cx.ntiles < omp_get_max_threads() ? cx.ntiles : omp_get_max_threads();
#pragma omp parallel num_threads(nthreads)
#endif
            {
#ifdef _OPENMP
                const int t = omp_get_thread_num();
#else
                const int t = 0;
#endif
                complex_t* const t1 = cx.tiles + 2*size_t(t)*cx.tile_size;
                complex_t* const t2 = t1 + cx.tile_size;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (int r0 = 0; r0 < R; r0 += C) {
                    for (int m = 0; m < M; m++)
                        std::memcpy(t1 + size_t(m)*C, x + size_t(R)*m + r0, C*sizeof(complex_t));
                    Kernel<M,C,inv>::run(t1, t2, cx.W1);
                    for (int k = 0; k < M; k++)
                        std::memcpy(y + size_t(k)*R + r0, t1 + size_t(k)*C, C*sizeof(complex_t));
                }
                // the implicit barrier of the loop above: y is complete

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (int k0 = 0; k0 < M; k0 += C) {
                    for (int b = 0; b < C; b++) {
                        const complex_t* const row = y + size_t(k0 + b)*R;
                        const long long k = k0 + b;
                        for (int r = 0; r < R; r++) {
                            const long long e = (static_cast<long long>(r) * k) & (N - 1);
                            complex_t w = cmul(cx.hi[e >> logm], cx.lo[e & (M - 1)]);
                            if (inv) w = cconj(w);
                            t1[b + size_t(C)*r] = cmul(row[r], w);
                        }
                    }
                    Kernel<R,C,inv>::run(t1, t2, cx.W2);
                    for (int q = 0; q < R; q++)
                        std::memcpy(x + size_t(k0) + size_t(q)*M, t1 + size_t(q)*C, C*sizeof(complex_t));
                }
            }
        }
    };

} /////////////////////////////////////////////////////////////////////////////

}

#endif // otfft_twophase_h
