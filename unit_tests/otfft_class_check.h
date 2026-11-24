// Checks of one algorithm class of one backend at a time (DIF8, DIT8, DIF16, DIT16, SixStep),
// for the tests that must reach code the table of createComplexFFT does not select
// (otfft_exp_test.cpp, otfft_sixstep_test.cpp).
//
// The classes are in the namespace of the backend, with the FFT_IF of that backend (the library
// includes otfft_if.h inside the namespace), so it is declared here the same way.
#pragma once

#include "otfft.h"

#include "otfft_test_checks.h"

#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef OTFFT_WITH_SSE2
namespace OTFFT_SSE2
{
#include "otfft_if.h"
    namespace OTFFT_AVXDIF8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_SixStep  { extern std::unique_ptr<FFT_IF> instance(); }
}
#endif
#ifdef OTFFT_WITH_AVX
namespace OTFFT_AVX
{
#include "otfft_if.h"
    namespace OTFFT_AVXDIF8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_SixStep  { extern std::unique_ptr<FFT_IF> instance(); }
}
#endif
#ifdef OTFFT_WITH_AVX2
namespace OTFFT_AVX2
{
#include "otfft_if.h"
    namespace OTFFT_AVXDIF8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_SixStep  { extern std::unique_ptr<FFT_IF> instance(); }
}
#endif

namespace OTFFT
{
namespace test
{
    // The three bins that are checked: 1, n/3 and n-1.
    inline int checkedBin(int index, int n)
    {
        return index == 0 ? 1 : (index == 1 ? n / 3 : n - 1);
    }

    // Unscaled direct sums for the three bins, sign -1 (forward) or +1 (inverse). Cached by
    // length and sign: they are the same for every class and backend. The signal is the one of
    // otfft_dft_check.h's makeSignal.
    inline const std::vector<complex_t>& classReference(int n, int sign)
    {
        static std::map<std::pair<int, int>, std::vector<complex_t>> cache;
        auto& entry = cache[std::make_pair(n, sign)];
        if (entry.empty())
        {
            const double pi = 3.14159265358979323846;
            std::vector<complex_t> input(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                input[static_cast<std::size_t>(i)] = complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
            }
            for (int b = 0; b < 3; ++b)
            {
                const long long k = checkedBin(b, n);
                double re = 0;
                double im = 0;
                for (long long j = 0; j < n; ++j)
                {
                    const double angle = sign * 2.0 * pi * static_cast<double>((j * k) & (n - 1)) / static_cast<double>(n);
                    const auto& v = input[static_cast<std::size_t>(j)];
                    re += v.Re * std::cos(angle) - v.Im * std::sin(angle);
                    im += v.Re * std::sin(angle) + v.Im * std::cos(angle);
                }
                entry.push_back(complex_t(re, im));
            }
        }
        return entry;
    }

    template <class F>
    void checkClassTransform(const F& f, void (F::*t)(complex_vector, complex_vector) const, int n, int sign, double scale)
    {
        std::vector<complex_t> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            x[static_cast<std::size_t>(i)] = complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        }
        simd_array<complex_t> work(n);
        (f.*t)(complex_vector{x.data()}, complex_vector{&work});
        const auto& ref = classReference(n, sign);
        for (int b = 0; b < 3; ++b)
        {
            CAPTURE(b);
            const complex_t& got = x[static_cast<std::size_t>(checkedBin(b, n))];
            CHECK_SMALL(got.Re - ref[static_cast<std::size_t>(b)].Re * scale, 1e-9 * n * scale);
            CHECK_SMALL(got.Im - ref[static_cast<std::size_t>(b)].Im * scale, 1e-9 * n * scale);
        }
    }

    // All six transform functions of one class instance for the length 2^log_n.
    template <class P>
    void checkClassLength(P f, int log_n)
    {
        typedef typename P::element_type F;
        const int n = 1 << log_n;
        f->setup(n);
        const double root = std::sqrt(static_cast<double>(n));
        checkClassTransform(*f, &F::fwd, n, -1, 1.0 / n);
        checkClassTransform(*f, &F::fwd0, n, -1, 1.0);
        checkClassTransform(*f, &F::fwdu, n, -1, 1.0 / root);
        checkClassTransform(*f, &F::inv, n, +1, 1.0);
        checkClassTransform(*f, &F::invu, n, +1, 1.0 / root);
        checkClassTransform(*f, &F::invn, n, +1, 1.0 / n);
    }

    template <typename Make>
    void checkClassLengths(const char* name, Make make, int min_log, int max_log)
    {
        CAPTURE(name);
        for (int log_n = min_log; log_n <= max_log; ++log_n)
        {
            CAPTURE(log_n);
            checkClassLength(make(), log_n);
        }
    }

    // Sets the OpenMP thread count for the scope of the object.
    struct ClassTestThreads
    {
#ifdef _OPENMP
        int saved = omp_get_max_threads();
        explicit ClassTestThreads(int n) { omp_set_num_threads(n); }
        ~ClassTestThreads() { omp_set_num_threads(saved); }
#else
        explicit ClassTestThreads(int) {}
#endif
    };
}
}
