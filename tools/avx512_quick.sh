#!/usr/bin/env bash
# Quick AVX-512 check of the OTFFT AVX-512 backend (roadmap R2.1, D32).
# Usage: tools/avx512_quick.sh [--full]   (--full also runs the whole unit test suite, which takes long)
# Needs cmake and g++ or clang++. Result: avx512-quick-result.txt in the repository root.
FULL=0; [ "${1:-}" = "--full" ] && FULL=1
cd "$(dirname "$0")/.." || exit 1
OUT=avx512-quick-result.txt
{
echo "== machine"; date; uname -srm
grep -m1 'model name' /proc/cpuinfo
echo "flags:"; grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|fma|avx512.*)$' | tr '\n' ' '; echo
g++ --version | head -1; git log --oneline -1
} | tee $OUT

echo "== build (unit tests + library only)" | tee -a $OUT
cmake -S . -B build-avx512 -DCMAKE_BUILD_TYPE=Release -DOTFFT_WITH_AVX512=ON \
      -DOTFFT_BUILD_EXAMPLES=OFF >/dev/null || { echo "CONFIGURE FAILED" | tee -a $OUT; exit 1; }
cmake --build build-avx512 --parallel 2 2>&1 | grep -E "warning|error|Error" | tee -a $OUT
TEST=$(find build-avx512 -type f -name 'otfft.*' -perm -u+x | grep -v '\.a$' | head -1)
echo "test binary: $TEST" | tee -a $OUT

echo "== AVX-512, backend and tuning tests" | tee -a $OUT
"$TEST" -tc="*vx512*,*Backend*,*Tuning*" 2>&1 | tail -5 | tee -a $OUT
if [ $FULL = 1 ]; then echo "== whole suite" | tee -a $OUT; "$TEST" 2>&1 | tail -4 | tee -a $OUT; fi

echo "== speed: AVX2 vs AVX-512 plan (about 1 minute)" | tee -a $OUT
cat > build-avx512/quick.cpp <<'CPP'
#include "otfft.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
using namespace OTFFT;
static double bestUs(ComplexFFTPtr& p, int n) {
    std::vector<complex_t> v(n);
    for (int i = 0; i < n; ++i) v[i] = complex_t((i * 7) % 11 - 5.0, (i * 5) % 13 - 6.0);
    double best = 1e30;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.15) {
        const auto a = std::chrono::steady_clock::now();
        p->fwd0(complex_vector{v.data()});
        best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count());
        for (int i = 0; i < n; ++i) { v[i].Re *= 1e-3; v[i].Im *= 1e-3; }   // keep values finite
    }
    return best;
}
int main() {
    const int sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 1000, 10000, 100000};
    printf("%9s %12s %12s %8s\n", "N", "avx2 us", "avx512 us", "ratio");
    for (int n : sizes) {
        auto a = Factory::createComplexFFT(n, OptimizationType::OPTIMIZED_FFT_AVX2);
        try {
            auto b = Factory::createComplexFFT(n, OptimizationType::OPTIMIZED_FFT_AVX512);
            double ta = bestUs(a, n), tb = bestUs(b, n), ta2 = bestUs(a, n), tb2 = bestUs(b, n);
            ta = std::min(ta, ta2); tb = std::min(tb, tb2);
            printf("%9d %12.1f %12.1f %8.2f\n", n, ta, tb, tb / ta);
        } catch (const std::exception& e) { printf("AVX-512 request failed: %s\n", e.what()); return 1; }
    }
}
CPP
LIB=$(find build-avx512 -name libotfft.a | head -1)
g++ -O2 -std=c++11 -fopenmp -Iinc -Isrc -Ibuild-avx512/src -Ibuild-avx512 build-avx512/quick.cpp "$LIB" -o build-avx512/quick 2>&1 | tail -5 | tee -a $OUT
build-avx512/quick 2>&1 | tee -a $OUT
echo; echo "Done. Send back: $OUT"
