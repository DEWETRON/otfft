# Quick AVX-512 check of the OTFFT AVX-512 backend (roadmap R2.1), Windows / MSVC.
# Usage: powershell -ExecutionPolicy Bypass -File tools\avx512_quick.ps1 [-Full]
# -Full also runs the whole unit test suite, which takes long. Needs cmake and Visual Studio (C++ tools).
# Result: avx512-quick-result-win.txt in the repository root.
param([switch]$Full)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$Out = Join-Path (Get-Location) "avx512-quick-result-win.txt"
function Log($s) { $s | Tee-Object -FilePath $Out -Append }
"" | Set-Content $Out

Log "== machine"
Log (Get-Date).ToString()
Log ((Get-CimInstance Win32_Processor).Name)
Log ("OS: " + (Get-CimInstance Win32_OperatingSystem).Caption)
Log ("commit: " + (git log --oneline -1))

Log "== build (library + unit tests)"
cmake -S . -B build-avx512 -DOTFFT_WITH_AVX512=ON -DOTFFT_BUILD_EXAMPLES=OFF 2>&1 | Select-Object -Last 3 | ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { Log "CONFIGURE FAILED"; exit 1 }
cmake --build build-avx512 --config Release --parallel 2 2>&1 | Select-String -Pattern "warning|error" | ForEach-Object { Log $_.Line }
if ($LASTEXITCODE -ne 0) { Log "BUILD FAILED (see lines above)"; }
$Test = Get-ChildItem -Recurse build-avx512 -Filter "otfft.*.exe" | Select-Object -First 1
if (-not $Test) { Log "NO TEST BINARY - build failed"; exit 1 }
Log ("test binary: " + $Test.FullName)

Log "== AVX-512, backend and tuning tests"
& $Test.FullName "-tc=*vx512*,*Backend*,*Tuning*" 2>&1 | Select-Object -Last 5 | ForEach-Object { Log $_ }
if ($Full) { Log "== whole suite"; & $Test.FullName 2>&1 | Select-Object -Last 4 | ForEach-Object { Log $_ } }

Log "== speed: AVX2 vs AVX-512 plan (about 1 minute)"
New-Item -ItemType Directory -Force build-quick | Out-Null
$repo = (Get-Location).Path -replace '\\','/'
@"
cmake_minimum_required(VERSION 3.10)
project(quick CXX)
set(OTFFT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OTFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OTFFT_WITH_AVX512 ON CACHE BOOL "" FORCE)
add_subdirectory($repo otfft)
add_executable(quick quick.cpp)
target_link_libraries(quick otfft)
"@ | Set-Content build-quick/CMakeLists.txt
@'
#include "otfft.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif
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
        for (int i = 0; i < n; ++i) { v[i].Re *= 1e-3; v[i].Im *= 1e-3; }
    }
    return best;
}
int main() {
#ifdef _MSC_VER
    int r[4]; __cpuidex(r, 7, 0);
    printf("CPUID 7: avx512f=%d avx512dq=%d avx512vl=%d\n", (r[1] >> 16) & 1, (r[1] >> 17) & 1, (r[1] >> 31) & 1);
#endif
    const int sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576, 1000, 10000, 100000};
    printf("%9s %12s %12s %8s\n", "N", "avx2 us", "avx512 us", "ratio");
    for (int n : sizes) {
        auto a = Factory::createComplexFFT(n, OptimizationType::OPTIMIZED_FFT_AVX2);
        try {
            auto b = Factory::createComplexFFT(n, OptimizationType::OPTIMIZED_FFT_AVX512);
            double ta = bestUs(a, n), tb = bestUs(b, n);
            ta = std::min(ta, bestUs(a, n)); tb = std::min(tb, bestUs(b, n));
            printf("%9d %12.1f %12.1f %8.2f\n", n, ta, tb, tb / ta);
        } catch (const std::exception& e) { printf("AVX-512 request failed: %s\n", e.what()); return 1; }
    }
}
'@ | Set-Content build-quick/quick.cpp
cmake -S build-quick -B build-quick/b 2>&1 | Select-Object -Last 2 | ForEach-Object { Log $_ }
cmake --build build-quick/b --config Release --parallel 2 2>&1 | Select-String -Pattern "error" | ForEach-Object { Log $_.Line }
$Q = Get-ChildItem -Recurse build-quick/b -Filter "quick.exe" | Select-Object -First 1
if ($Q) { & $Q.FullName 2>&1 | ForEach-Object { Log $_ } } else { Log "quick.exe was not built" }
Log ""; Log "Done. Send back: $Out"
