// The TwoPhase algorithm (roadmap R1.x, decision D35, spec CPX-20 to CPX-22): a threaded complex
// FFT in two parallel phases for the lengths 2^16 to 2^22, built from the kernels of the DIF8
// class. The table of createComplexFFT does not select it unless a tuning file or the built-in
// table says so, so these tests call the class of each backend directly. The strips of each phase
// are split over the threads with a static schedule, so 3 and 8 threads give an uneven split.
#include "otfft_class_check.h"

#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

// the algorithm a new plan of length 2^log_N gets with the current thread count (src/otfft_algo.cpp)
#ifdef OTFFT_WITH_SSE2
namespace OTFFT_SSE2 { int selectedAlgorithm(int log_N) noexcept; }
#endif
#ifdef OTFFT_WITH_AVX
namespace OTFFT_AVX { int selectedAlgorithm(int log_N) noexcept; }
#endif
#ifdef OTFFT_WITH_AVX2
namespace OTFFT_AVX2 { int selectedAlgorithm(int log_N) noexcept; }
#endif

#ifdef OTFFT_WITH_AVX512
namespace OTFFT_AVX512
{
    int selectedAlgorithm(int log_N) noexcept;
#include "otfft_if.h"
    namespace OTFFT_AVXDIF8  { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_TwoPhase { extern std::unique_ptr<FFT_IF> instance(); }
}
#endif

namespace
{
    using OTFFT::complex_t;
    using OTFFT::complex_vector;
    using OTFFT::simd_array;
    using OTFFT::test::checkClassLengths;
    using OTFFT::test::ClassTestThreads;

    // three bins against the direct sum, all six functions (otfft_class_check.h)
    void checkBinsAllBackends(int min_log, int max_log)
    {
#ifdef OTFFT_WITH_SSE2
        checkClassLengths("OTFFT_SSE2 TwoPhase", OTFFT_SSE2::OTFFT_TwoPhase::instance, min_log, max_log);
#endif
#ifdef OTFFT_WITH_AVX
        checkClassLengths("OTFFT_AVX TwoPhase", OTFFT_AVX::OTFFT_TwoPhase::instance, min_log, max_log);
#endif
#ifdef OTFFT_WITH_AVX2
        checkClassLengths("OTFFT_AVX2 TwoPhase", OTFFT_AVX2::OTFFT_TwoPhase::instance, min_log, max_log);
#endif
#ifdef OTFFT_WITH_AVX512
        checkClassLengths("OTFFT_AVX512 TwoPhase", OTFFT_AVX512::OTFFT_TwoPhase::instance, min_log, max_log);
#endif
    }

    std::vector<complex_t> randomSignal(int n)
    {
        std::mt19937_64 rng(1234 + n);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<complex_t> v(static_cast<std::size_t>(n));
        for (auto& c : v)
        {
            c = complex_t(dist(rng), dist(rng));
        }
        return v;
    }

    // Every value of the output of the TwoPhase class against the DIF8 class (the table of the
    // library uses it, and the bins check above ties it to the direct sum): the relative error
    // in the 2-norm of the whole array.
    template <class Make2, class Make8>
    void compareWholeArrays(const char* name, Make2 makeTwoPhase, Make8 makeDif8, int log_n)
    {
        CAPTURE(name);
        CAPTURE(log_n);
        const int n = 1 << log_n;
        auto two = makeTwoPhase();
        auto ref = makeDif8();
        two->setup(n);
        ref->setup(n);
        const std::vector<complex_t> in = randomSignal(n);

        typedef typename decltype(two)::element_type F;
        typedef void (F::*Function)(complex_vector, complex_vector) const;
        const Function funcs[6] = {&F::fwd, &F::fwd0, &F::fwdu, &F::inv, &F::invu, &F::invn};
        const char* const names[6] = {"fwd", "fwd0", "fwdu", "inv", "invu", "invn"};
        for (int f = 0; f < 6; ++f)
        {
            CAPTURE(names[f]);
            std::vector<complex_t> a = in;
            std::vector<complex_t> b = in;
            simd_array<complex_t> w1(n);
            simd_array<complex_t> w2(n);
            ((*two).*funcs[f])(complex_vector{a.data()}, complex_vector{&w1});
            ((*ref).*funcs[f])(complex_vector{b.data()}, complex_vector{&w2});
            double num = 0;
            double den = 0;
            for (int i = 0; i < n; ++i)
            {
                const double dr = a[static_cast<std::size_t>(i)].Re - b[static_cast<std::size_t>(i)].Re;
                const double di = a[static_cast<std::size_t>(i)].Im - b[static_cast<std::size_t>(i)].Im;
                num += dr * dr + di * di;
                den += b[static_cast<std::size_t>(i)].Re * b[static_cast<std::size_t>(i)].Re +
                       b[static_cast<std::size_t>(i)].Im * b[static_cast<std::size_t>(i)].Im;
            }
            CHECK_SMALL(std::sqrt(num / den), 1e-13);
        }
    }

    // A plan keeps working when the thread count changes after setup: made with `planThreads`, run with
    // `runThreads` (more threads than the plan has tiles must not overrun them, fewer must be honoured).
    template <class Make2, class Make8>
    void planThenChangeThreads(const char* name, Make2 makeTwoPhase, Make8 makeDif8, int log_n, int planThreads, int runThreads)
    {
        CAPTURE(name);
        CAPTURE(planThreads);
        CAPTURE(runThreads);
        const int n = 1 << log_n;
        auto two = makeTwoPhase();
        auto ref = makeDif8();
        {
            ClassTestThreads threads(planThreads);
            two->setup(n);
        }
        ref->setup(n);
        const std::vector<complex_t> in = randomSignal(n);
        std::vector<complex_t> a = in;
        std::vector<complex_t> b = in;
        simd_array<complex_t> w1(n);
        simd_array<complex_t> w2(n);
        {
            ClassTestThreads threads(runThreads);
            two->fwd0(complex_vector{a.data()}, complex_vector{&w1});
            two->invn(complex_vector{a.data()}, complex_vector{&w1});   // back to the input
        }
        ref->fwd0(complex_vector{b.data()}, complex_vector{&w2});
        ref->invn(complex_vector{b.data()}, complex_vector{&w2});
        double num = 0;
        double den = 0;
        for (int i = 0; i < n; ++i)
        {
            const double dr = a[static_cast<std::size_t>(i)].Re - in[static_cast<std::size_t>(i)].Re;
            const double di = a[static_cast<std::size_t>(i)].Im - in[static_cast<std::size_t>(i)].Im;
            num += dr * dr + di * di;
            den += in[static_cast<std::size_t>(i)].Re * in[static_cast<std::size_t>(i)].Re +
                   in[static_cast<std::size_t>(i)].Im * in[static_cast<std::size_t>(i)].Im;
        }
        CHECK_SMALL(std::sqrt(num / den), 1e-13);
    }

    void compareAllBackends(int log_n)
    {
#ifdef OTFFT_WITH_SSE2
        compareWholeArrays("OTFFT_SSE2", OTFFT_SSE2::OTFFT_TwoPhase::instance, OTFFT_SSE2::OTFFT_AVXDIF8::instance, log_n);
#endif
#ifdef OTFFT_WITH_AVX
        compareWholeArrays("OTFFT_AVX", OTFFT_AVX::OTFFT_TwoPhase::instance, OTFFT_AVX::OTFFT_AVXDIF8::instance, log_n);
#endif
#ifdef OTFFT_WITH_AVX2
        compareWholeArrays("OTFFT_AVX2", OTFFT_AVX2::OTFFT_TwoPhase::instance, OTFFT_AVX2::OTFFT_AVXDIF8::instance, log_n);
#endif
#ifdef OTFFT_WITH_AVX512
        compareWholeArrays("OTFFT_AVX512", OTFFT_AVX512::OTFFT_TwoPhase::instance, OTFFT_AVX512::OTFFT_AVXDIF8::instance, log_n);
#endif
    }

    const char* const kFile = "otfft_twophase_test_file.txt";

    struct TuningGuard
    {
        TuningGuard() { OTFFT::resetTuning(); }
        ~TuningGuard()
        {
            OTFFT::resetTuning();
            std::remove(kFile);
        }
    };

    bool loadTuning(const std::string& text, std::string& error)
    {
        {
            std::ofstream out(kFile);
            out << text;
        }
        return OTFFT::loadTuningFile(kFile, error);
    }

    // a tuning file that gives TwoPhase to all its lengths in the threaded table of the backend
    std::string twoPhaseTuning(const char* backend)
    {
        std::string text = "otfft-tuning 1\n";
        for (int log_n = 16; log_n <= 22; ++log_n)
        {
            text += std::string(backend) + " threaded " + std::to_string(log_n) + " TwoPhase\n";
        }
        return text;
    }

    // The complex plan, the real plan and the DCT plan of createComplexFFT, createRealFFT and
    // createDCT with TwoPhase selected by a tuning file, against the plans of the built-in tables.
    // The buffers are SIMD-aligned (spec FAC-7); the spectrum of the real FFT gets room for n + 2 values.
    struct Buffers
    {
        simd_array<complex_t> complexData;
        simd_array<double> realIn;
        simd_array<complex_t> spectrum;
        simd_array<double> dct;
        explicit Buffers(int n, const std::vector<complex_t>& in) : complexData(n), realIn(n), spectrum(n + 2), dct(n)
        {
            for (int i = 0; i < n; ++i)
            {
                complexData[i] = in[static_cast<std::size_t>(i)];
                realIn[i] = in[static_cast<std::size_t>(i)].Re;
                dct[i] = in[static_cast<std::size_t>(i)].Re;
            }
        }
        void run(int n, OTFFT::OptimizationType type)
        {
            OTFFT::Factory::createComplexFFT(n, type)->fwd0(complex_vector{&complexData});
            OTFFT::Factory::createRealFFT(n, type)->fwd0(OTFFT::const_double_vector{&realIn}, complex_vector{&spectrum});
            OTFFT::Factory::createDCT(n, type)->fwd0(OTFFT::double_vector{&dct});
        }
    };

    double relativeError(const simd_array<complex_t>& a, const simd_array<complex_t>& b, int n)
    {
        double num = 0;
        double den = 0;
        for (int i = 0; i < n; ++i)
        {
            const double dr = a[i].Re - b[i].Re;
            const double di = a[i].Im - b[i].Im;
            num += dr * dr + di * di;
            den += b[i].Re * b[i].Re + b[i].Im * b[i].Im;
        }
        return std::sqrt(num / den);
    }

    void compareFactory(OTFFT::OptimizationType type, const char* backend)
    {
        CAPTURE(backend);
        TuningGuard guard;
        ClassTestThreads threads(4);
        const int n = 1 << 18;
        const std::vector<complex_t> in = randomSignal(n);

        Buffers ref(n, in);
        ref.run(n, type);

        std::string error;
        REQUIRE_MESSAGE(loadTuning(twoPhaseTuning(backend), error), error);
        Buffers got(n, in);
        got.run(n, type);

        CHECK_SMALL(relativeError(got.complexData, ref.complexData, n), 1e-13);
        CHECK_SMALL(relativeError(got.spectrum, ref.spectrum, n / 2 + 1), 1e-13);
        double num = 0;
        double den = 0;
        for (int i = 0; i < n; ++i)
        {
            num += (got.dct[i] - ref.dct[i]) * (got.dct[i] - ref.dct[i]);
            den += ref.dct[i] * ref.dct[i];
        }
        CHECK_SMALL(std::sqrt(num / den), 1e-13);
    }
}

TEST_SUITE("otfft_twophase_test")
{

TEST_CASE("TestTwoPhaseBinsOneThread")
{
    ClassTestThreads threads(1);
    checkBinsAllBackends(16, 22);
}

TEST_CASE("TestTwoPhaseBinsThreeThreads")
{
    ClassTestThreads threads(3);
    checkBinsAllBackends(16, 22);
}

TEST_CASE("TestTwoPhaseBinsEightThreads")
{
    ClassTestThreads threads(8);
    checkBinsAllBackends(16, 22);
}

TEST_CASE("TestTwoPhaseWholeArraySmallLengthsFiveThreads")
{
    ClassTestThreads threads(5);
    compareAllBackends(16);
    compareAllBackends(17);
    compareAllBackends(19);
}

TEST_CASE("TestTwoPhaseWholeArrayLargestLengthEightThreads")
{
    ClassTestThreads threads(8);
    compareAllBackends(22);
}

TEST_CASE("TestTwoPhasePlanSurvivesThreadCountChange")
{
    // made with 8 threads and run with 1 and 3; made with 2 and run with 16 (more than the tiles)
#ifdef OTFFT_WITH_AVX2
    planThenChangeThreads("OTFFT_AVX2", OTFFT_AVX2::OTFFT_TwoPhase::instance, OTFFT_AVX2::OTFFT_AVXDIF8::instance, 18, 8, 1);
    planThenChangeThreads("OTFFT_AVX2", OTFFT_AVX2::OTFFT_TwoPhase::instance, OTFFT_AVX2::OTFFT_AVXDIF8::instance, 18, 8, 3);
    planThenChangeThreads("OTFFT_AVX2", OTFFT_AVX2::OTFFT_TwoPhase::instance, OTFFT_AVX2::OTFFT_AVXDIF8::instance, 18, 2, 16);
#endif
#ifdef OTFFT_WITH_SSE2
    planThenChangeThreads("OTFFT_SSE2", OTFFT_SSE2::OTFFT_TwoPhase::instance, OTFFT_SSE2::OTFFT_AVXDIF8::instance, 17, 8, 1);
    planThenChangeThreads("OTFFT_SSE2", OTFFT_SSE2::OTFFT_TwoPhase::instance, OTFFT_SSE2::OTFFT_AVXDIF8::instance, 17, 2, 16);
#endif
}

TEST_CASE("TestTwoPhaseDefaultTable")
{
    // The built-in table (no tuning file) gives TwoPhase (number 8) to the AVX2 and AVX-512 backends:
    // from 2^18 to 2^22 with 2 or more threads, 2^16 and 2^17 from 8 threads; never with one thread,
    // never outside 2^16 to 2^22, and never to the SSE2 and AVX backends.
    TuningGuard guard;
    struct Case { int threads; int log_n; bool twoPhase; };
    const Case cases[] = {
        {16, 16, true},  {16, 17, true},  {16, 20, true},  {16, 22, true},  {16, 15, false}, {16, 23, false}, {16, 24, false},
        {8, 16, true},   {8, 17, true},   {8, 18, true},
        {4, 16, false},  {4, 17, false},  {4, 18, true},   {4, 22, true},
        {2, 16, false},  {2, 17, false},  {2, 18, true},   {2, 21, true},
        {1, 16, false},  {1, 20, false},  {1, 22, false}};
#ifdef _OPENMP
    for (const Case& c : cases)
    {
        ClassTestThreads threads(c.threads);
        CAPTURE(c.threads);
        CAPTURE(c.log_n);
#ifdef OTFFT_WITH_AVX2
        CHECK((OTFFT_AVX2::selectedAlgorithm(c.log_n) == 8) == c.twoPhase);
#endif
#ifdef OTFFT_WITH_AVX512
        CHECK((OTFFT_AVX512::selectedAlgorithm(c.log_n) == 8) == c.twoPhase);
#endif
#ifdef OTFFT_WITH_AVX
        CHECK(OTFFT_AVX::selectedAlgorithm(c.log_n) != 8);
#endif
#ifdef OTFFT_WITH_SSE2
        CHECK(OTFFT_SSE2::selectedAlgorithm(c.log_n) != 8);
#endif
    }
#endif
}

TEST_CASE("TestTwoPhaseTuningFileWinsOverDefaultTable")
{
#if defined(OTFFT_WITH_AVX2) && defined(_OPENMP)
    TuningGuard guard;
    ClassTestThreads threads(16);
    std::string error;
    REQUIRE_MESSAGE(loadTuning("otfft-tuning 1\navx2 threaded 20 DIF8\n", error), error);
    CHECK(OTFFT_AVX2::selectedAlgorithm(20) == 3);   // DIF8 from the file
    CHECK(OTFFT_AVX2::selectedAlgorithm(19) == 8);   // the default table
#endif
}

TEST_CASE("TestTwoPhaseOtherLengthsAreRefused")
{
    // the class exists for 2^16 to 2^22 only; the tuning file loader and the table refuse the rest
#ifdef OTFFT_WITH_AVX2
    for (int log_n : {1, 12, 15, 23, 24})
    {
        CAPTURE(log_n);
        auto f = OTFFT_AVX2::OTFFT_TwoPhase::instance();
        CHECK_THROWS_AS(f->setup(1 << log_n), std::domain_error);
    }
#endif
    TuningGuard guard;
    std::string error;
    CHECK_FALSE(loadTuning("otfft-tuning 1\navx2 threaded 12 TwoPhase\n", error));
    CHECK(error.find("TwoPhase") != std::string::npos);
    CHECK_FALSE(loadTuning("otfft-tuning 1\navx2 serial 23 TwoPhase\n", error));
    CHECK(loadTuning("otfft-tuning 1\navx2 threaded 16 TwoPhase\navx2 serial 22 TwoPhase\n", error));
}

#ifdef OTFFT_WITH_SSE2
TEST_CASE("TestTwoPhaseSelectedByTuningFileSse2")
{
    compareFactory(OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2, "sse2");
}
#endif
#ifdef OTFFT_WITH_AVX
TEST_CASE("TestTwoPhaseSelectedByTuningFileAvx")
{
    compareFactory(OTFFT::OptimizationType::OPTIMIZED_FFT_AVX, "avx");
}
#endif
#ifdef OTFFT_WITH_AVX2
TEST_CASE("TestTwoPhaseSelectedByTuningFileAvx2")
{
    compareFactory(OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2, "avx2");
}
#endif

}
