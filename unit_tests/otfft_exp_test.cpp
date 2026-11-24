// The exp kernels of the radix-8 and radix-16 classes (roadmap R1.9, decision D19): from
// N = 2^18 the classes DIF8, DIT8, DIF16 and DIT16 compute the twiddle factors in registers
// instead of reading a weight table. The table of createComplexFFT picks one class per length,
// so this test calls each class of each backend directly. It checks all six transform
// functions (scales 1/N, 1, 1/sqrt(N)) against the direct sum, for lengths 2^18 to 2^21 (all
// remainders of log2(N) modulo 3 and 4), with one thread and with four.
#include "otfft.h"
#include "otfft_test_checks.h"

#include <cmath>
#include <map>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// The classes are in the namespace of the backend, with the FFT_IF of that backend (the library
// includes otfft_if.h inside the namespace), so it is declared here the same way.
#ifdef OTFFT_WITH_SSE2
namespace OTFFT_SSE2
{
#include "otfft_if.h"
    namespace OTFFT_AVXDIF8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT16 { extern std::unique_ptr<FFT_IF> instance(); }
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
}
#endif

namespace
{
    using OTFFT::complex_t;

    std::vector<complex_t> makeSignal(int n)
    {
        std::vector<complex_t> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            x[static_cast<std::size_t>(i)] = complex_t(((i * 7) % 11) - 5.0, ((i * 5) % 13) - 6.0);
        }
        return x;
    }

    // The three bins that are checked: 1, n/3 and n-1.
    int binOf(int index, int n)
    {
        return index == 0 ? 1 : (index == 1 ? n / 3 : n - 1);
    }

    // Unscaled direct sums for the three bins, sign -1 (forward) or +1 (inverse). Cached by
    // length and sign: they are the same for every class and backend.
    const std::vector<complex_t>& reference(int n, int sign)
    {
        static std::map<std::pair<int, int>, std::vector<complex_t>> cache;
        auto& entry = cache[std::make_pair(n, sign)];
        if (entry.empty())
        {
            const double pi = 3.14159265358979323846;
            const auto input = makeSignal(n);
            for (int b = 0; b < 3; ++b)
            {
                const long long k = binOf(b, n);
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
    void checkTransform(const F& f, void (F::*t)(OTFFT::complex_vector, OTFFT::complex_vector) const, int n, int sign, double scale)
    {
        std::vector<complex_t> x = makeSignal(n);
        OTFFT::simd_array<complex_t> work(n);
        (f.*t)(OTFFT::complex_vector{x.data()}, OTFFT::complex_vector{&work});
        const auto& ref = reference(n, sign);
        for (int b = 0; b < 3; ++b)
        {
            CAPTURE(b);
            const complex_t& got = x[static_cast<std::size_t>(binOf(b, n))];
            CHECK_SMALL(got.Re - ref[static_cast<std::size_t>(b)].Re * scale, 1e-9 * n * scale);
            CHECK_SMALL(got.Im - ref[static_cast<std::size_t>(b)].Im * scale, 1e-9 * n * scale);
        }
    }

    template <class P>
    void checkClass(P f, int log_n)
    {
        typedef typename P::element_type F;
        const int n = 1 << log_n;
        f->setup(n);
        const double root = std::sqrt(static_cast<double>(n));
        checkTransform(*f, &F::fwd, n, -1, 1.0 / n);
        checkTransform(*f, &F::fwd0, n, -1, 1.0);
        checkTransform(*f, &F::fwdu, n, -1, 1.0 / root);
        checkTransform(*f, &F::inv, n, +1, 1.0);
        checkTransform(*f, &F::invu, n, +1, 1.0 / root);
        checkTransform(*f, &F::invn, n, +1, 1.0 / n);
    }

    template <typename Make>
    void checkLengths(const char* name, Make make)
    {
        CAPTURE(name);
        for (int log_n = 18; log_n <= 21; ++log_n)
        {
            CAPTURE(log_n);
            checkClass(make(), log_n);
        }
    }

    struct ThreadCount
    {
#ifdef _OPENMP
        int saved = omp_get_max_threads();
        explicit ThreadCount(int n) { omp_set_num_threads(n); }
        ~ThreadCount() { omp_set_num_threads(saved); }
#else
        explicit ThreadCount(int) {}
#endif
    };

#define OTFFT_EXP_TEST_BACKEND(NS)                                                   \
    checkLengths(#NS " DIF8", NS::OTFFT_AVXDIF8::instance);                          \
    checkLengths(#NS " DIT8", NS::OTFFT_AVXDIT8::instance);                          \
    checkLengths(#NS " DIF16", NS::OTFFT_AVXDIF16::instance);                        \
    checkLengths(#NS " DIT16", NS::OTFFT_AVXDIT16::instance);

    void checkAllBackends()
    {
#ifdef OTFFT_WITH_SSE2
        OTFFT_EXP_TEST_BACKEND(OTFFT_SSE2)
#endif
#ifdef OTFFT_WITH_AVX
        OTFFT_EXP_TEST_BACKEND(OTFFT_AVX)
#endif
#ifdef OTFFT_WITH_AVX2
        OTFFT_EXP_TEST_BACKEND(OTFFT_AVX2)
#endif
    }
}

TEST_SUITE("otfft_exp_test")
{

TEST_CASE("TestExpKernelsOneThread")
{
    ThreadCount threads(1);
    checkAllBackends();
}

TEST_CASE("TestExpKernelsFourThreads")
{
    ThreadCount threads(4);
    checkAllBackends();
}

}
