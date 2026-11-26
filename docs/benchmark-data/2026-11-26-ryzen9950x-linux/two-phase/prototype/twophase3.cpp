// Experiment (roadmap R1.x, FFTW comparison): a two-phase threaded complex FFT built from the
// existing OTFFT AVX2 kernels, laid out like the plans FFTW prints for 16 threads:
//   N = R * M, M = 2048 fixed.  x[n], n = R*m + r.
//   Phase 1: R child FFTs of size M (stride R in the input), one strip of C children per work item,
//            each strip gathered into a cache-resident tile, transformed completely, scattered to Y[k][r].
//   Phase 2: for strips of B columns k: twiddle W_N^(r k), R-point FFT across r, scatter to x[k + q M].
// Memory traffic: read x, write Y, read Y, write x = two read+write sweeps.
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"
#include "otfft_avx2.h"
#include "otfft_avxdif8.h"

#include <fftw3.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>
static bool THP = false;
static int PAD = 0; // extra complex values at the end of each row of the work array Y (breaks power-of-two strides)
static void* bigalloc(size_t bytes) { size_t a = 2u << 20; bytes = (bytes + a - 1) / a * a; void* p = aligned_alloc(THP ? a : 64, bytes); if (THP) madvise(p, bytes, MADV_HUGEPAGE); return p; }

namespace K = OTFFT_TP::OTFFT_AVXDIF8;
using OTFFT::complex_t;


static int ABL = 0; // bit0: skip phase-1 FFT, bit1: skip phase-2 FFT, bit2: skip twiddle (copy only)
static inline complex_t cmul(complex_t a, complex_t b) { return complex_t(a.Re * b.Re - a.Im * b.Im, a.Re * b.Im + a.Im * b.Re); }

struct Tile { complex_t* t1; complex_t* t2; };

struct TwoPhaseBase { virtual ~TwoPhaseBase() {} virtual void fwd0(complex_t* x, complex_t* y) const = 0; virtual bool ok() const = 0; };

template <int LOGM, int C>
class TwoPhase : public TwoPhaseBase
{
    static const int M = 1 << LOGM, B = C;
public:
    bool ok() const override { return m_ok; }
    explicit TwoPhase(int logN) : m_logN(logN), m_N(1 << logN), m_R(m_N / M)
    {
        m_ok = (logN - LOGM >= 3) && (logN - LOGM <= 13) && ((size_t)m_R * B <= (1u << 15));
        if (!m_ok) return;
        const int T = M;                      // twiddle table split: e = ehi*T + elo
        m_lo.resize(T); m_hi.resize(m_N / T);
        for (int i = 0; i < T; ++i) m_lo[i] = unit(i, m_N);
        for (int i = 0; i < m_N / T; ++i) m_hi[i] = unit((long long)i * T, m_N);
        m_w1.setup(2 * M * C);
        m_w1p = &m_w1;
        K::init_Wt(8, M * C, m_w1p);
        m_w2.setup(2 * m_R * B);
        m_w2p = &m_w2;
        K::init_Wt(8, m_R * B, m_w2p);
        const int nt = omp_get_max_threads();
        const size_t tile = (size_t)std::max(M * C, m_R * B);
        for (int i = 0; i < nt; ++i)
        {
            Tile t;
            t.t1 = (complex_t*)aligned_alloc(64, tile * sizeof(complex_t));
            t.t2 = (complex_t*)aligned_alloc(64, tile * sizeof(complex_t));
            m_tiles.push_back(t);
        }
    }
    ~TwoPhase() override { for (auto& t : m_tiles) { free(t.t1); free(t.t2); } }

    // x: input and output (in place), y: work array of N points
    void fwd0(complex_t* x, complex_t* y) const override
    {
        const int R = m_R, strips1 = R / C, blocks2 = M / B;
        double ta = 0, tb = 0, tc = 0;
#pragma omp parallel
        {
            const Tile tl = m_tiles[omp_get_thread_num()];
            if (omp_get_thread_num() == 0) ta = omp_get_wtime();
#pragma omp for schedule(static)
            for (int s = 0; s < strips1; ++s)
            {
                const int r0 = s * C;
                for (int m = 0; m < M; ++m)
                    std::memcpy(tl.t1 + (size_t)m * C, x + (size_t)R * m + r0, C * sizeof(complex_t));
                if (!(ABL & 1)) K::fwdfft<M, C, false, OTFFT_MISC::scale_1>()(tl.t1, tl.t2, m_w1p);
                for (int k = 0; k < M; ++k)
                    std::memcpy(y + (size_t)k * (R + PAD) + r0, tl.t1 + (size_t)k * C, C * sizeof(complex_t));
            }
            // implicit barrier: all of y is written before phase 2 reads it
            if (omp_get_thread_num() == 0) tb = omp_get_wtime();
#pragma omp for schedule(static)
            for (int bi = 0; bi < blocks2; ++bi)
            {
                const int k0 = bi * B;
                for (int b = 0; b < B; ++b)
                {
                    const complex_t* row = y + (size_t)(k0 + b) * (R + PAD);
                    const long long k = k0 + b;
                    for (int r = 0; r < R; ++r)
                    {
                        const long long e = ((long long)r * k) & (m_N - 1);
                        tl.t1[b + (size_t)B * r] = (ABL & 4) ? row[r] : cmul(row[r], cmul(m_hi[e / M], m_lo[e % M]));
                    }
                }
                if (!(ABL & 2)) dispatch2(tl.t1, tl.t2);
                for (int q = 0; q < R; ++q)
                    std::memcpy(x + (size_t)k0 + (size_t)q * M, tl.t1 + (size_t)q * B, B * sizeof(complex_t));
            }
            if (omp_get_thread_num() == 0) tc = omp_get_wtime();
        }
        m_p1 += (tb - ta) * 1e6; m_p2 += (tc - tb) * 1e6; m_calls++;
    }
    mutable double m_p1 = 0, m_p2 = 0; mutable long m_calls = 0;
    void report(const char* tag) const { printf("   %s phase 1 %.1f us, phase 2 %.1f us (thread 0 view, mean over %ld calls)\n", tag, m_p1 / m_calls, m_p2 / m_calls, m_calls); }

private:
    static complex_t unit(long long e, int N)
    {
        const long double th = -2.0L * 3.14159265358979323846264338327950288L * (long double)e / (long double)N;
        return complex_t((double)cosl(th), (double)sinl(th));
    }
    void dispatch2(complex_t* a, complex_t* b) const
    {
        using namespace OTFFT_MISC;
        switch (m_logN - LOGM)
        {
        case 3:  K::fwdfft<(1 << 3),  B, false, scale_1>()(a, b, m_w2p); break;
        case 4:  K::fwdfft<(1 << 4),  B, false, scale_1>()(a, b, m_w2p); break;
        case 5:  K::fwdfft<(1 << 5),  B, false, scale_1>()(a, b, m_w2p); break;
        case 6:  K::fwdfft<(1 << 6),  B, false, scale_1>()(a, b, m_w2p); break;
        case 7:  K::fwdfft<(1 << 7),  B, false, scale_1>()(a, b, m_w2p); break;
        case 8:  K::fwdfft<(1 << 8),  B, false, scale_1>()(a, b, m_w2p); break;
        case 9:  K::fwdfft<(1 << 9),  B, false, scale_1>()(a, b, m_w2p); break;
        case 10: K::fwdfft<(1 << 10), B, false, scale_1>()(a, b, m_w2p); break;
        case 11: K::fwdfft<(1 << 11), B, false, scale_1>()(a, b, m_w2p); break;
        case 12: K::fwdfft<(1 << 12), B, false, scale_1>()(a, b, m_w2p); break;
        case 13: K::fwdfft<(1 << 13), B, false, scale_1>()(a, b, m_w2p); break;
        default: std::abort();
        }
    }
    bool m_ok = true;
    int m_logN, m_N, m_R;
    std::vector<complex_t> m_lo, m_hi;
    OTFFT::simd_array<complex_t> m_w1, m_w2;
    complex_t* m_w1p; complex_t* m_w2p;
    std::vector<Tile> m_tiles;
};

static double nowUs() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class F> static double medianUs(F&& f, int minReps, double minTotalUs)
{
    std::vector<double> t;
    double total = 0;
    for (int i = 0; i < 3; ++i) f();                 // warm up
    while ((int)t.size() < minReps || total < minTotalUs)
    {
        const double a = nowUs(); f(); const double d = nowUs() - a;
        t.push_back(d); total += d;
        if (t.size() > 100000) break;
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}


struct Cfg { const char* name; TwoPhaseBase* (*make)(int); };
template <int LOGM, int C> static TwoPhaseBase* mk(int logN) { return new TwoPhase<LOGM, C>(logN); }
static const Cfg CFGS[] = {
    {"M512 C4",   mk<9, 4>},   {"M1024 C4", mk<10, 4>}, {"M2048 C4", mk<11, 4>}, {"M4096 C4", mk<12, 4>},
    {"M512 C8",   mk<9, 8>},   {"M1024 C8", mk<10, 8>}, {"M2048 C8", mk<11, 8>},
    {"M256 C16",  mk<8, 16>},  {"M512 C16", mk<9, 16>}, {"M1024 C16", mk<10, 16>},
};

int main(int argc, char** argv)
{
    const int threads = argc > 1 ? atoi(argv[1]) : 16;
    THP = getenv("THP") != nullptr;
    if (getenv("PAD")) PAD = atoi(getenv("PAD"));
    omp_set_num_threads(threads);
    fftw_init_threads();
    fftw_plan_with_nthreads(threads);
    fftw_set_timelimit(10.0);
    printf("threads %d; microseconds per forward transform (median); two-phase configs: child size M, strip width C\n", threads);
    printf("%9s %10s %10s |", "N", "OTFFT-nc", "FFTW");
    for (const Cfg& c : CFGS) printf(" %9s", c.name);
    printf("\n");
    for (int logN = 16; logN <= 22; ++logN)
    {
        const int N = 1 << logN;
        std::vector<double> in(2 * (size_t)N);
        unsigned long long s = 88172645463325252ULL;
        for (auto& v : in) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v = ((s >> 11) * (1.0 / 9007199254740992.0)) * 2 - 1; }
        complex_t* x = (complex_t*)bigalloc((size_t)N * 16);
        complex_t* y = (complex_t*)bigalloc(((size_t)N + (size_t)8192 * 16) * 16);
        complex_t* ref = (complex_t*)bigalloc((size_t)N * 16);
        complex_t* x0 = (complex_t*)bigalloc((size_t)N * 16);
        std::memcpy(x0, in.data(), (size_t)N * 16);
        auto lib = OTFFT::Factory::createComplexFFT(N, OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2);
        std::memcpy(ref, x0, (size_t)N * 16);
        lib->fwd0(ref);
        auto timeReload = [&](auto&& body) {
            std::vector<double> t; double total = 0;
            for (int i = 0; i < 3; ++i) { std::memcpy(x, x0, (size_t)N * 16); body(); }
            while (t.size() < 21 || total < 200000) {
                std::memcpy(x, x0, (size_t)N * 16);
                const double a = nowUs(); body(); const double d = nowUs() - a; t.push_back(d); total += d;
                if (t.size() > 20000) break;
            }
            std::sort(t.begin(), t.end()); return t[t.size() / 2];
        };
        const double tLib = timeReload([&] { lib->fwd0(x); });
        fftw_complex* fin = (fftw_complex*)bigalloc(sizeof(fftw_complex) * N);
        fftw_complex* fout = (fftw_complex*)bigalloc(sizeof(fftw_complex) * N);
        std::memcpy(fin, x0, (size_t)N * 16);
        fftw_plan p = fftw_plan_dft_1d(N, fin, fout, FFTW_FORWARD, FFTW_MEASURE);
        std::memcpy(fin, x0, (size_t)N * 16);
        const double tFftw = medianUs([&] { fftw_execute(p); }, 21, 200000);
        fftw_destroy_plan(p);
        printf("%9d %10.1f %10.1f |", N, tLib, tFftw);
        for (const Cfg& c : CFGS)
        {
            std::unique_ptr<TwoPhaseBase> tp(c.make(logN));
            if (!tp->ok()) { printf(" %9s", "-"); continue; }
            std::memcpy(x, x0, (size_t)N * 16);
            tp->fwd0(x, y);
            double num = 0, den = 0;
            for (int i = 0; i < N; ++i) { const double dr = x[i].Re - ref[i].Re, di = x[i].Im - ref[i].Im; num += dr * dr + di * di; den += ref[i].Re * ref[i].Re + ref[i].Im * ref[i].Im; }
            if (std::sqrt(num / den) > 1e-12) { printf(" %9s", "WRONG"); continue; }
            printf(" %9.1f", timeReload([&] { tp->fwd0(x, y); }));
        }
        printf("\n"); fflush(stdout);
        free(x); free(y); free(ref); free(x0);
    }
}
