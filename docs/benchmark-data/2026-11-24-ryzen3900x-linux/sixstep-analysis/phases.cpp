// Analysis harness (not part of the repo): times the five passes of the square six-step
// code (otfft_sixstepsq.h part2/part3) per thread count and schedule.
#include "/home/glaure/Development/otfft/src/otfft_algo_sixstep.cpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace OTFFT_NAMESPACE;
using namespace OTFFT_NAMESPACE::OTFFT_SixStep;
typedef std::chrono::steady_clock Clock;
static Clock::time_point T0 = Clock::now();
static double now_us() { return std::chrono::duration<double, std::micro>(Clock::now() - T0).count(); }

template <int LOG>
void run(int reps, const char* sched)
{
    typedef fwdffts<LOG, scale_1> S;
    const int N = 1 << LOG;
    OTFFT_AVX2::OTFFT_SixStep::FFT0 f; f.setup2(LOG);
    simd_array<complex_t> ax(N), ay(N);
    complex_t* x = &ax; complex_t* y = &ay;
    for (int i = 0; i < N; i++) { x[i].Re = std::cos(0.001 * i); x[i].Im = std::sin(0.002 * i); }
    const index_t* iv = f.iv; const complex_t* W = f.W; const complex_t* Ws = f.Ws;
    const int n = S::n, L = S::L;
    std::vector<std::vector<double>> per(6);
    for (int r = 0; r < reps + 2; r++) {
        double t[6];
        #pragma omp parallel
        {
            #pragma omp barrier
            #pragma omp master
            t[0] = now_us();
            #pragma omp for schedule(runtime)
            for (int i = 0; i < L; i++) S::transpose_kernel(iv[i].row, iv[i].col, x);
            #pragma omp master
            t[1] = now_us();
            #pragma omp for schedule(runtime)
            for (int p = 0; p < n; p++) OTFFT_FourStep::fwdfft<S::n,1,0,scale_1>()(x + p*n, y + p*n, Ws);
            #pragma omp master
            t[2] = now_us();
            #pragma omp for schedule(runtime)
            for (int i = 0; i < L; i++) S::mult_twiddle_factor_kernel(i, iv[i].row, iv[i].col, x, W);
            #pragma omp master
            t[3] = now_us();
            #pragma omp for schedule(runtime)
            for (int k = 0; k < n; k++) OTFFT_FourStep::fwdfft<S::n,1,0,scale_1>()(x + k*n, y + k*n, Ws);
            #pragma omp master
            t[4] = now_us();
            #pragma omp for schedule(runtime)
            for (int i = 0; i < L; i++) S::transpose_kernel(iv[i].row, iv[i].col, x);
            #pragma omp master
            t[5] = now_us();
        }
        if (r >= 2) { for (int k = 0; k < 5; k++) per[k].push_back(t[k+1]-t[k]); per[5].push_back(t[5]-t[0]); }
    }
    auto med = [](std::vector<double> v){ std::sort(v.begin(), v.end()); return v[v.size()/2]; };
    std::printf("log2N=%d threads=%d sched=%s  transpose1 %.0f | rowFFT1 %.0f | twiddle+tr %.0f | rowFFT2 %.0f | transpose2 %.0f | total %.0f us\n",
        LOG, omp_get_max_threads(), sched, med(per[0]), med(per[1]), med(per[2]), med(per[3]), med(per[4]), med(per[5]));
}

int main(int argc, char** argv)
{
    int lg = argc > 1 ? std::atoi(argv[1]) : 18;
    const char* sched = argc > 2 ? argv[2] : "static";
    int reps = argc > 3 ? std::atoi(argv[3]) : 41;
    if (!std::strcmp(sched, "guided")) omp_set_schedule(omp_sched_guided, 0); else omp_set_schedule(omp_sched_static, 0);
    switch (lg) { case 16: run<16>(reps, sched); break; case 18: run<18>(reps, sched); break; case 20: run<20>(reps, sched); break; case 22: run<22>(reps, sched); break; default: return 1; }
    return 0;
}
