// Cost of a sweep over data that another thread wrote in the previous sweep. Not part of the repo.
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
    int shift = argc > 2 ? atoi(argv[2]) : 0;
    size_t n = mib * 1024 * 1024 / 32;
    __m256d* x = (__m256d*)aligned_alloc(64, n * 32);
    int T = omp_get_max_threads();
    size_t chunk = n / T;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) x[i] = _mm256_set1_pd(1.0);
    std::vector<double> t2;
    for (int r = 0; r < 60; r++) {
        double a = 0, b = 0;
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            #pragma omp barrier
            auto t0 = C::now();
            for (size_t i = t * chunk; i < (t + 1) * chunk; i++) x[i] = _mm256_add_pd(x[i], _mm256_set1_pd(1e-9));
            #pragma omp barrier
            auto t1 = C::now();
            int c = (t + shift) % T;
            for (size_t i = c * chunk; i < (c + 1) * chunk; i++) x[i] = _mm256_add_pd(x[i], _mm256_set1_pd(1e-9));
            #pragma omp barrier
            auto t2_ = C::now();
            #pragma omp master
            { a = std::chrono::duration<double,std::micro>(t1-t0).count(); b = std::chrono::duration<double,std::micro>(t2_-t1).count(); }
        }
        if (r >= 10) t2.push_back(b);
    }
    std::sort(t2.begin(), t2.end());
    double bytes = 2.0 * chunk * T * 32;
    std::printf("array %zu MiB threads=%d shift=%d: sweep of another thread's data %.0f us (%.0f GB/s)\n", mib, T, shift, t2[t2.size()/2], bytes/t2[t2.size()/2]/1e3);
}
