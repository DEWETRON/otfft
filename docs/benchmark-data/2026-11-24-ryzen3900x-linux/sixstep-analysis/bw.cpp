// Parallel bandwidth ceiling: in-place read+write sweep of an array (like a transpose pass) and
// a copy x -> y, per thread count. Not part of the repo.
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <omp.h>
typedef std::chrono::steady_clock C;
int main(int argc, char** argv)
{
    size_t mib = argc > 1 ? atoi(argv[1]) : 4;
    size_t n = mib * 1024 * 1024 / 32; // __m256d count
    __m256d* x = (__m256d*)aligned_alloc(64, n * 32);
    __m256d* y = (__m256d*)aligned_alloc(64, n * 32);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { x[i] = _mm256_set1_pd(1.0); y[i] = _mm256_set1_pd(2.0); }
    std::vector<double> sweep, copy;
    for (int r = 0; r < 60; r++) {
        auto t0 = C::now();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) x[i] = _mm256_add_pd(x[i], _mm256_set1_pd(1e-9));  // read + write
        auto t1 = C::now();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) y[i] = x[i];  // read x, write y
        auto t2 = C::now();
        if (r >= 10) { sweep.push_back(std::chrono::duration<double,std::micro>(t1-t0).count()); copy.push_back(std::chrono::duration<double,std::micro>(t2-t1).count()); }
    }
    std::sort(sweep.begin(), sweep.end()); std::sort(copy.begin(), copy.end());
    double bytes = 2.0 * n * 32;
    std::printf("array %zu MiB threads=%d: in-place sweep %.0f us (%.0f GB/s) | copy %.0f us (%.0f GB/s)\n", mib, omp_get_max_threads(),
        sweep[sweep.size()/2], bytes/sweep[sweep.size()/2]/1e3, copy[copy.size()/2], bytes/copy[copy.size()/2]/1e3);
}
