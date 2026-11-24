// Library fwd0 (one thread) with y placed at a chosen byte offset behind x. Not part of the repo.
#include "/home/glaure/Development/otfft/src/otfft_algo_sixstep.cpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace OTFFT_AVX2;
typedef std::chrono::steady_clock C;
int main(int argc, char** argv)
{
    const int LOG = 18, N = 1 << LOG;
    long off = argc > 1 ? std::atol(argv[1]) : 0;            // bytes between the end of x and the start of y
    OTFFT_AVX2::OTFFT_SixStep::FFT0 f; f.setup2(LOG);
    size_t bytes = (size_t)N * 16;
    char* base = (char*)aligned_alloc(1 << 21, 4 * bytes + (1 << 22));   // 2 MiB aligned
    complex_t* x = (complex_t*)base;
    complex_t* y = (complex_t*)(base + bytes + off);
    std::vector<double> w;
    for (int r = 0; r < 41; r++) {
        for (int i = 0; i < N; i++) { x[i].Re = std::cos(0.001 * i); x[i].Im = std::sin(0.002 * i); }
        auto t0 = C::now(); f.fwd0(x, y); auto t1 = C::now();
        if (r >= 3) w.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(w.begin(), w.end());
    std::printf("offset %8ld bytes (y - end of x): fwd0 %.0f us\n", off, w[w.size() / 2]);
}
