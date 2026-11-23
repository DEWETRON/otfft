// OTFFT benchmark: compares OTFFT with FFTW, PocketFFT and KissFFT.
//
// What is timed (see docs/benchmarks.md):
//   * one forward, unnormalised transform of a given input, plan already built
//   * libraries that work in place (OTFFT complex and DCT) also pay for copying
//     the input into their work buffer, because the input would be destroyed
//     otherwise. Out-of-place libraries do not copy.
// Every result is checked against PocketFFT before it is timed; a library that
// gets a wrong answer for a case is reported as invalid and not timed.

#include "otfft.h"

#include "kiss_fft.h"
#include "kiss_fftr.h"

#include "pocketfft_hdronly.h"

#ifdef HAVE_FFTW
#include <fftw3.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{

// RoundTrip: complex forward + inverse of the same data, as in the benchmark
// on the upstream OTFFT page (cost = DFT + IDFT).
enum class Kind { Complex, Real, DCT, RoundTrip };

const char* kindName(Kind k)
{
    switch (k)
    {
    case Kind::Complex: return "complex";
    case Kind::Real: return "real";
    case Kind::DCT: return "dct";
    case Kind::RoundTrip: return "roundtrip";
    }
    return "?";
}

// Number of doubles in the input and in the output of a transform.
std::size_t inputDoubles(Kind k, std::size_t n)
{
    return (k == Kind::Complex || k == Kind::RoundTrip) ? 2 * n : n;
}
std::size_t outputDoubles(Kind k, std::size_t n)
{
    switch (k)
    {
    case Kind::Complex:
    case Kind::RoundTrip: return 2 * n;
    case Kind::Real: return 2 * (n / 2 + 1);
    case Kind::DCT: return n;
    }
    return 0;
}

// Conventional operation counts (as in benchFFT), so MFLOPS are comparable.
double equivalentFlops(Kind k, std::size_t n)
{
    const double nl = static_cast<double>(n) * std::log2(static_cast<double>(n));
    switch (k)
    {
    case Kind::Complex: return 5.0 * nl;
    case Kind::RoundTrip: return 10.0 * nl;
    default: return 2.5 * nl;
    }
}

template <class T>
struct AlignedBuf
{
    T* p = nullptr;
    AlignedBuf() = default;
    explicit AlignedBuf(std::size_t count) { reset(count); }
    ~AlignedBuf() { release(); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    void reset(std::size_t count)
    {
        release();
        std::size_t bytes = (count * sizeof(T) + 63) / 64 * 64 + 64;
#ifdef _MSC_VER
        p = static_cast<T*>(_aligned_malloc(bytes, 64));
#else
        void* mem = nullptr;
        if (posix_memalign(&mem, 64, bytes) != 0)
        {
            mem = nullptr;
        }
        p = static_cast<T*>(mem);
#endif
        std::memset(p, 0, bytes);
    }
    void release()
    {
#ifdef _MSC_VER
        _aligned_free(p);
#else
        std::free(p);
#endif
        p = nullptr;
    }
    T* get() const { return p; }
};

// ---------------------------------------------------------------------------
// Adapters: one per library (and per OTFFT backend)
// ---------------------------------------------------------------------------

class Adapter
{
public:
    virtual ~Adapter() = default;
    virtual std::string name() const = 0;
    virtual bool supports(Kind k, std::size_t n) const = 0;
    // Build the plan and buffers. Must not be called twice without teardown().
    virtual void setup(Kind k, std::size_t n) = 0;
    virtual void teardown() = 0;
    // Copy the input (not timed). `in` has inputDoubles() values.
    virtual void load(const double* in) = 0;
    // One transform of the loaded input.
    virtual void run() = 0;
    // Copy the output out. `out` has outputDoubles() values.
    virtual void result(double* out) = 0;
    // Number of threads the library actually uses with the requested count.
    virtual int effectiveThreads(int requested) const { (void)requested; return 1; }
};

// ----- OTFFT --------------------------------------------------------------

class OtfftAdapter : public Adapter
{
public:
    OtfftAdapter(std::string label, OTFFT::OptimizationType type, bool bluestein, bool threaded)
        : m_label(std::move(label)), m_type(type), m_bluestein(bluestein), m_threaded(threaded) {}

    std::string name() const override { return m_label; }

    bool supports(Kind k, std::size_t n) const override
    {
        if (m_bluestein && (k != Kind::Complex || (n & (n - 1)) == 0))
        {
            return false; // Bluestein is for lengths the mixed radix does not like
        }
        if (k != Kind::Complex && (n % 2) != 0)
        {
            return false; // OTFFT real transforms need an even length
        }
        return n >= 2;
    }

    void setup(Kind k, std::size_t n) override
    {
        m_kind = k;
        m_n = n;
        const int len = static_cast<int>(n);
        switch (k)
        {
        case Kind::Complex:
        case Kind::RoundTrip:
            m_cfft = m_bluestein ? OTFFT::Factory::createBluesteinFFT(len, m_type)
                                 : OTFFT::Factory::createComplexFFT(len, m_type);
            m_in.reset(2 * n);
            m_work.reset(2 * n);
            break;
        case Kind::Real:
            m_rfft = OTFFT::Factory::createRealFFT(len, m_type);
            m_in.reset(n);
            m_work.reset(2 * (n + 2)); // spectrum, N/2+1 complex values
            break;
        case Kind::DCT:
            m_dct = OTFFT::Factory::createDCT(len, m_type);
            m_in.reset(n);
            m_work.reset(n);
            break;
        }
    }

    void teardown() override
    {
        m_cfft.reset();
        m_rfft.reset();
        m_dct.reset();
    }

    void load(const double* in) override
    {
        std::memcpy(m_in.get(), in, inputDoubles(m_kind, m_n) * sizeof(double));
        if (m_kind == Kind::RoundTrip)
        {
            std::memcpy(m_work.get(), in, 2 * m_n * sizeof(double));
        }
    }

    void run() override
    {
        switch (m_kind)
        {
        case Kind::RoundTrip:
            // fwd scales by 1/N and inv does not: the data comes back unchanged,
            // so it can be transformed in place again and again without copying
            m_cfft->fwd(reinterpret_cast<OTFFT::complex_t*>(m_work.get()));
            m_cfft->inv(reinterpret_cast<OTFFT::complex_t*>(m_work.get()));
            break;
        case Kind::Complex:
            std::memcpy(m_work.get(), m_in.get(), 2 * m_n * sizeof(double));
            m_cfft->fwd0(reinterpret_cast<OTFFT::complex_t*>(m_work.get()));
            break;
        case Kind::Real:
            m_rfft->fwd0(m_in.get(), reinterpret_cast<OTFFT::complex_t*>(m_work.get()));
            break;
        case Kind::DCT:
            std::memcpy(m_work.get(), m_in.get(), m_n * sizeof(double));
            m_dct->fwd0(m_work.get());
            break;
        }
    }

    void result(double* out) override
    {
        std::memcpy(out, m_work.get(), outputDoubles(m_kind, m_n) * sizeof(double));
    }

    int effectiveThreads(int requested) const override { return m_threaded ? requested : 1; }

private:
    std::string m_label;
    OTFFT::OptimizationType m_type;
    bool m_bluestein;
    bool m_threaded;
    Kind m_kind = Kind::Complex;
    std::size_t m_n = 0;
    AlignedBuf<double> m_in, m_work;
    OTFFT::ComplexFFTPtr m_cfft{nullptr};
    OTFFT::RealFFTPtr m_rfft{nullptr};
    OTFFT::RealDCTPtr m_dct{nullptr};
};

// ----- PocketFFT (also the reference) ---------------------------------------

class PocketAdapter : public Adapter
{
public:
    std::string name() const override { return "PocketFFT"; }
    bool supports(Kind k, std::size_t n) const override
    {
        (void)k;
        return n >= 2;
    }
    void setup(Kind k, std::size_t n) override
    {
        m_kind = k;
        m_n = n;
        m_in.reset(inputDoubles(k, n));
        m_out.reset(outputDoubles(k, n));
        m_out2.reset(outputDoubles(k, n));
        // warm the plan cache so that planning is not part of the first run
        run();
    }
    void teardown() override {}
    void load(const double* in) override
    {
        std::memcpy(m_in.get(), in, inputDoubles(m_kind, m_n) * sizeof(double));
    }
    void run() override
    {
        using namespace pocketfft;
        const shape_t shape{m_n};
        switch (m_kind)
        {
        case Kind::RoundTrip:
            c2c(shape, stride_t{sizeof(std::complex<double>)}, stride_t{sizeof(std::complex<double>)},
                shape_t{0}, FORWARD, reinterpret_cast<const std::complex<double>*>(m_in.get()),
                reinterpret_cast<std::complex<double>*>(m_out.get()), 1.0);
            c2c(shape, stride_t{sizeof(std::complex<double>)}, stride_t{sizeof(std::complex<double>)},
                shape_t{0}, BACKWARD, reinterpret_cast<const std::complex<double>*>(m_out.get()),
                reinterpret_cast<std::complex<double>*>(m_out2.get()), 1.0);
            break;
        case Kind::Complex:
            c2c(shape, stride_t{sizeof(std::complex<double>)}, stride_t{sizeof(std::complex<double>)},
                shape_t{0}, FORWARD, reinterpret_cast<const std::complex<double>*>(m_in.get()),
                reinterpret_cast<std::complex<double>*>(m_out.get()), 1.0);
            break;
        case Kind::Real:
            r2c(shape, stride_t{sizeof(double)}, stride_t{sizeof(std::complex<double>)}, 0, FORWARD,
                m_in.get(), reinterpret_cast<std::complex<double>*>(m_out.get()), 1.0);
            break;
        case Kind::DCT:
            dct(shape, stride_t{sizeof(double)}, stride_t{sizeof(double)}, shape_t{0}, 2, m_in.get(),
                m_out.get(), 1.0, false);
            break;
        }
    }
    void result(double* out) override
    {
        const double* src = m_kind == Kind::RoundTrip ? m_out2.get() : m_out.get();
        std::memcpy(out, src, outputDoubles(m_kind, m_n) * sizeof(double));
    }

private:
    Kind m_kind = Kind::Complex;
    std::size_t m_n = 0;
    AlignedBuf<double> m_in, m_out, m_out2;
};

// ----- KissFFT --------------------------------------------------------------

class KissAdapter : public Adapter
{
public:
    std::string name() const override { return "KissFFT"; }
    bool supports(Kind k, std::size_t n) const override
    {
        if (k == Kind::DCT)
        {
            return false;
        }
        if (k == Kind::Real && (n % 2) != 0)
        {
            return false;
        }
        return n >= 2;
    }
    void setup(Kind k, std::size_t n) override
    {
        m_kind = k;
        m_n = n;
        m_in.reset(inputDoubles(k, n));
        m_out.reset(outputDoubles(k, n));
        if (k == Kind::Complex || k == Kind::RoundTrip)
        {
            m_cfg = kiss_fft_alloc(static_cast<int>(n), 0, nullptr, nullptr);
            m_icfg = kiss_fft_alloc(static_cast<int>(n), 1, nullptr, nullptr);
            m_out2.reset(2 * n);
        }
        else
        {
            m_rcfg = kiss_fftr_alloc(static_cast<int>(n), 0, nullptr, nullptr);
        }
    }
    void teardown() override
    {
        std::free(m_cfg);
        std::free(m_icfg);
        std::free(m_rcfg);
        m_icfg = nullptr;
        m_cfg = nullptr;
        m_rcfg = nullptr;
    }
    void load(const double* in) override
    {
        std::memcpy(m_in.get(), in, inputDoubles(m_kind, m_n) * sizeof(double));
    }
    void run() override
    {
        if (m_kind == Kind::RoundTrip)
        {
            kiss_fft(m_cfg, reinterpret_cast<const kiss_fft_cpx*>(m_in.get()),
                     reinterpret_cast<kiss_fft_cpx*>(m_out.get()));
            kiss_fft(m_icfg, reinterpret_cast<const kiss_fft_cpx*>(m_out.get()),
                     reinterpret_cast<kiss_fft_cpx*>(m_out2.get()));
        }
        else if (m_kind == Kind::Complex)
        {
            kiss_fft(m_cfg, reinterpret_cast<const kiss_fft_cpx*>(m_in.get()),
                     reinterpret_cast<kiss_fft_cpx*>(m_out.get()));
        }
        else
        {
            kiss_fftr(m_rcfg, m_in.get(), reinterpret_cast<kiss_fft_cpx*>(m_out.get()));
        }
    }
    void result(double* out) override
    {
        const double* src = m_kind == Kind::RoundTrip ? m_out2.get() : m_out.get();
        std::memcpy(out, src, outputDoubles(m_kind, m_n) * sizeof(double));
    }

private:
    Kind m_kind = Kind::Complex;
    std::size_t m_n = 0;
    AlignedBuf<double> m_in, m_out, m_out2;
    kiss_fft_cfg m_cfg = nullptr;
    kiss_fft_cfg m_icfg = nullptr;
    kiss_fftr_cfg m_rcfg = nullptr;
};

// ----- FFTW -----------------------------------------------------------------

#ifdef HAVE_FFTW
class FftwAdapter : public Adapter
{
public:
    FftwAdapter(std::string label, unsigned flags, bool threaded)
        : m_label(std::move(label)), m_flags(flags), m_threaded(threaded) {}

    std::string name() const override { return m_label; }
    bool supports(Kind k, std::size_t n) const override
    {
        (void)k;
        return n >= 2;
    }
    void setup(Kind k, std::size_t n) override
    {
        m_kind = k;
        m_n = n;
        m_in = static_cast<double*>(fftw_malloc(inputDoubles(k, n) * sizeof(double)));
        m_out = static_cast<double*>(fftw_malloc(outputDoubles(k, n) * sizeof(double)));
        std::memset(m_in, 0, inputDoubles(k, n) * sizeof(double));
        std::memset(m_out, 0, outputDoubles(k, n) * sizeof(double));
        const int len = static_cast<int>(n);
#ifdef HAVE_FFTW_OMP
        fftw_plan_with_nthreads(m_threaded ? s_threads : 1);
#endif
        switch (k)
        {
        case Kind::RoundTrip:
            m_out2 = static_cast<double*>(fftw_malloc(2 * n * sizeof(double)));
            std::memset(m_out2, 0, 2 * n * sizeof(double));
            m_plan = fftw_plan_dft_1d(len, reinterpret_cast<fftw_complex*>(m_in),
                                      reinterpret_cast<fftw_complex*>(m_out), FFTW_FORWARD, m_flags);
            m_plan2 = fftw_plan_dft_1d(len, reinterpret_cast<fftw_complex*>(m_out),
                                       reinterpret_cast<fftw_complex*>(m_out2), FFTW_BACKWARD, m_flags);
            break;
        case Kind::Complex:
            m_plan = fftw_plan_dft_1d(len, reinterpret_cast<fftw_complex*>(m_in),
                                      reinterpret_cast<fftw_complex*>(m_out), FFTW_FORWARD, m_flags);
            break;
        case Kind::Real:
            m_plan = fftw_plan_dft_r2c_1d(len, m_in, reinterpret_cast<fftw_complex*>(m_out), m_flags);
            break;
        case Kind::DCT:
            m_plan = fftw_plan_r2r_1d(len, m_in, m_out, FFTW_REDFT10, m_flags);
            break;
        }
        if (!m_plan || (k == Kind::RoundTrip && !m_plan2))
        {
            std::fprintf(stderr, "FFTW could not plan %s n=%zu\n", kindName(k), n);
            std::exit(2);
        }
    }
    void teardown() override
    {
        if (m_plan)
        {
            fftw_destroy_plan(m_plan);
        }
        if (m_plan2)
        {
            fftw_destroy_plan(m_plan2);
        }
        m_plan = nullptr;
        m_plan2 = nullptr;
        fftw_free(m_in);
        fftw_free(m_out);
        fftw_free(m_out2);
        m_in = m_out = m_out2 = nullptr;
    }
    void load(const double* in) override
    {
        std::memcpy(m_in, in, inputDoubles(m_kind, m_n) * sizeof(double));
    }
    void run() override
    {
        fftw_execute(m_plan);
        if (m_plan2)
        {
            fftw_execute(m_plan2);
        }
    }
    void result(double* out) override
    {
        const double* src = m_kind == Kind::RoundTrip ? m_out2 : m_out;
        std::memcpy(out, src, outputDoubles(m_kind, m_n) * sizeof(double));
    }
    int effectiveThreads(int requested) const override
    {
#ifdef HAVE_FFTW_OMP
        return m_threaded ? requested : 1;
#else
        (void)requested;
        return 1;
#endif
    }

    static int s_threads;

private:
    std::string m_label;
    unsigned m_flags;
    bool m_threaded;
    Kind m_kind = Kind::Complex;
    std::size_t m_n = 0;
    double* m_in = nullptr;
    double* m_out = nullptr;
    double* m_out2 = nullptr;
    fftw_plan m_plan = nullptr;
    fftw_plan m_plan2 = nullptr;
};
int FftwAdapter::s_threads = 1;
#endif

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

double nowNs()
{
    return std::chrono::duration<double, std::nano>(Clock::now().time_since_epoch()).count();
}

struct Options
{
    std::vector<Kind> kinds{Kind::Complex, Kind::Real, Kind::DCT};
    std::vector<std::string> categories{"pow2", "smooth", "prime"};
    std::string libFilter;
    int threads = 1;
    int samples = 11;
    double targetMs = 2.0;
    double maxRunMs = 2000.0;
    std::string csvPath;
    bool quick = false;
};

struct Size
{
    std::string category;
    std::size_t n;
};

std::vector<Size> sizesFor(const Options& o)
{
    std::vector<Size> v;
    auto want = [&](const std::string& c) {
        return std::find(o.categories.begin(), o.categories.end(), c) != o.categories.end();
    };
    if (want("pow2"))
    {
        const int step = o.quick ? 4 : 2;
        for (int e = 4; e <= (o.quick ? 20 : 22); e += step)
        {
            v.push_back({"pow2", std::size_t(1) << e});
        }
    }
    if (want("pow2all"))
    {
        // every exponent, like the table on the upstream OTFFT page
        for (int e = 1; e <= 24; ++e)
        {
            v.push_back({"pow2all", std::size_t(1) << e});
        }
    }
    if (want("smooth"))
    {
        for (std::size_t n : {480u, 1000u, 3000u, 10080u, 100000u, 1000000u})
        {
            v.push_back({"smooth", n});
        }
    }
    if (want("prime"))
    {
        for (std::size_t n : {1009u, 10007u, 100003u})
        {
            v.push_back({"prime", n});
        }
    }
    return v;
}

std::vector<std::unique_ptr<Adapter>> makeAdapters(const Options& o)
{
    std::vector<std::unique_ptr<Adapter>> a;
    a.emplace_back(new OtfftAdapter("OTFFT", OTFFT::OptimizationType::OPTIMIZED_FFT_AUTO, false, true));
#ifdef OTFFT_WITH_SSE2
    a.emplace_back(new OtfftAdapter("OTFFT-sse2", OTFFT::OptimizationType::OPTIMIZED_FFT_SSE2, false, true));
#endif
#ifdef OTFFT_WITH_AVX
    a.emplace_back(new OtfftAdapter("OTFFT-avx", OTFFT::OptimizationType::OPTIMIZED_FFT_AVX, false, true));
#endif
#ifdef OTFFT_WITH_AVX2
    a.emplace_back(new OtfftAdapter("OTFFT-avx2", OTFFT::OptimizationType::OPTIMIZED_FFT_AVX2, false, true));
#endif
    a.emplace_back(new OtfftAdapter("OTFFT-bluestein", OTFFT::OptimizationType::OPTIMIZED_FFT_AUTO, true, true));
#ifdef HAVE_FFTW
    a.emplace_back(new FftwAdapter("FFTW-measure", FFTW_MEASURE, true));
    a.emplace_back(new FftwAdapter("FFTW-estimate", FFTW_ESTIMATE, true));
#endif
    a.emplace_back(new PocketAdapter());
    a.emplace_back(new KissAdapter());

    if (!o.libFilter.empty())
    {
        a.erase(std::remove_if(a.begin(), a.end(),
                               [&](const std::unique_ptr<Adapter>& p) {
                                   return p->name().find(o.libFilter) == std::string::npos &&
                                          p->name() != "PocketFFT"; // always needed as reference
                               }),
                a.end());
    }
    return a;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Compare `got` with `ref`. Returns the relative error after the best scalar
// fit; `scale` is that fit (1.0 when the conventions agree).
double compare(const std::vector<double>& got, const std::vector<double>& ref, double& scale)
{
    double gr = 0, rr = 0, maxRef = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        gr += got[i] * ref[i];
        rr += ref[i] * ref[i];
        maxRef = std::max(maxRef, std::fabs(ref[i]));
    }
    scale = rr > 0 ? gr / rr : 1.0;
    if (std::fabs(scale - 1.0) < 1e-9)
    {
        scale = 1.0;
    }
    double maxErr = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        maxErr = std::max(maxErr, std::fabs(got[i] - scale * ref[i]));
    }
    return maxRef > 0 ? maxErr / maxRef : maxErr;
}

std::string cpuModel()
{
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
    {
        if (line.compare(0, 10, "model name") == 0)
        {
            auto pos = line.find(':');
            return pos == std::string::npos ? line : line.substr(pos + 2);
        }
    }
    return "unknown";
}

std::string readFirstLine(const char* path)
{
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

void printEnvironment(const Options& o)
{
    std::fprintf(stderr, "# cpu: %s\n", cpuModel().c_str());
    std::fprintf(stderr, "# governor: %s\n",
                 readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").c_str());
#ifdef __VERSION__
    std::fprintf(stderr, "# compiler: %s\n", __VERSION__);
#endif
#ifdef __OPTIMIZE__
    std::fprintf(stderr, "# optimised build: yes\n");
#else
    std::fprintf(stderr, "# optimised build: NO (results are not meaningful)\n");
#endif
#ifdef HAVE_FFTW
    std::fprintf(stderr, "# fftw: %s\n", fftw_version);
#endif
    std::fprintf(stderr, "# threads: %d, samples: %d, target batch: %.1f ms\n", o.threads, o.samples,
                 o.targetMs);
}

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
    {
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

bool parseArgs(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--threads") o.threads = std::max(1, std::atoi(next().c_str()));
        else if (a == "--lib") o.libFilter = next();
        else if (a == "--csv") o.csvPath = next();
        else if (a == "--samples") o.samples = std::max(3, std::atoi(next().c_str()));
        else if (a == "--target-ms") o.targetMs = std::atof(next().c_str());
        else if (a == "--max-run-ms") o.maxRunMs = std::atof(next().c_str());
        else if (a == "--quick") { o.quick = true; o.samples = 5; o.targetMs = 1.0; }
        else if (a == "--sizes") o.categories = split(next(), ',');
        else if (a == "--kinds")
        {
            o.kinds.clear();
            for (const auto& k : split(next(), ','))
            {
                if (k == "complex") o.kinds.push_back(Kind::Complex);
                else if (k == "real") o.kinds.push_back(Kind::Real);
                else if (k == "dct") o.kinds.push_back(Kind::DCT);
                else if (k == "roundtrip") o.kinds.push_back(Kind::RoundTrip);
                else { std::fprintf(stderr, "unknown kind %s\n", k.c_str()); return false; }
            }
        }
        else
        {
            std::fprintf(stderr,
                         "usage: otfft_bench [--kinds complex,real,dct,roundtrip] [--sizes pow2,pow2all,smooth,prime]\n"
                         "                   [--lib NAME] [--threads T] [--samples S] [--target-ms MS]\n                   [--max-run-ms MS]\n"
                         "                   [--csv FILE] [--quick]\n");
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Options o;
    if (!parseArgs(argc, argv, o))
    {
        return 1;
    }

#ifdef _OPENMP
    omp_set_num_threads(o.threads);
#endif
#ifdef HAVE_FFTW_OMP
    fftw_init_threads();
    FftwAdapter::s_threads = o.threads;
#endif
#ifdef HAVE_FFTW
    fftw_set_timelimit(10.0);
#endif

    printEnvironment(o);

    std::FILE* csv = stdout;
    if (!o.csvPath.empty())
    {
        csv = std::fopen(o.csvPath.c_str(), "w");
        if (!csv)
        {
            std::perror("csv");
            return 1;
        }
    }
    std::fprintf(csv, "library,kind,category,n,threads,median_ns,min_ns,mflops,plan_us,rel_error,scale,status\n");

    auto adapters = makeAdapters(o);
    PocketAdapter* reference = nullptr;
    for (auto& a : adapters)
    {
        if (auto* p = dynamic_cast<PocketAdapter*>(a.get()))
        {
            reference = p;
        }
    }

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (Kind kind : o.kinds)
    {
        for (const Size& sz : sizesFor(o))
        {
            const std::size_t n = sz.n;
            if (kind != Kind::Complex && (n % 2) != 0)
            {
                // still run: libraries that support odd real lengths are shown; others are skipped
            }
            std::vector<double> input(inputDoubles(kind, n));
            for (auto& x : input)
            {
                x = dist(rng);
            }

            // reference result
            std::vector<double> ref(outputDoubles(kind, n));
            if (kind == Kind::RoundTrip)
            {
                ref = input; // forward + inverse gives the input back, up to a scale
            }
            else
            {
                reference->setup(kind, n);
                reference->load(input.data());
                reference->run();
                reference->result(ref.data());
                reference->teardown();
            }

            for (auto& a : adapters)
            {
                if (!a->supports(kind, n))
                {
                    continue;
                }
                // single-threaded libraries give the same numbers at any thread count
                if (o.threads > 1 && a->effectiveThreads(o.threads) == 1)
                {
                    continue;
                }

                // plan
                const double p0 = nowNs();
                a->setup(kind, n);
                const double planUs = (nowNs() - p0) / 1e3;

                a->load(input.data());
                a->run();
                std::vector<double> got(outputDoubles(kind, n));
                a->result(got.data());
                double scale = 1.0;
                const double err = compare(got, ref, scale);
                // unnormalised round trips accumulate more rounding error with N
                const bool valid = std::isfinite(err) && err < (kind == Kind::RoundTrip ? 1e-7 : 1e-9);
                // a different DCT scale is fine, a different complex/real scale is not
                const bool scaleOk = kind == Kind::DCT || kind == Kind::RoundTrip || scale == 1.0;

                if (!valid || !scaleOk)
                {
                    std::fprintf(csv, "%s,%s,%s,%zu,%d,,,,%.1f,%.3g,%.6g,invalid\n", a->name().c_str(),
                                 kindName(kind), sz.category.c_str(), n, a->effectiveThreads(o.threads),
                                 planUs, err, scale);
                    std::fflush(csv);
                    a->teardown();
                    continue;
                }

                // time one run: skip hopeless cases, use fewer samples for slow ones
                const double w0 = nowNs();
                a->run();
                const double oneRunNs = nowNs() - w0;
                if (oneRunNs > o.maxRunMs * 1e6)
                {
                    std::fprintf(csv, "%s,%s,%s,%zu,%d,,,,%.1f,%.3g,%.6g,too_slow\n", a->name().c_str(),
                                 kindName(kind), sz.category.c_str(), n, a->effectiveThreads(o.threads),
                                 planUs, err, scale);
                    std::fflush(csv);
                    a->teardown();
                    continue;
                }
                const int samples = oneRunNs > 200e6 ? std::max(3, o.samples / 3) : o.samples;
                for (int i = 0; i < 2; ++i)
                {
                    a->run();
                }
                std::size_t reps = 1;
                for (;;)
                {
                    const double t0 = nowNs();
                    for (std::size_t r = 0; r < reps; ++r)
                    {
                        a->run();
                    }
                    const double dt = nowNs() - t0;
                    if (dt >= o.targetMs * 1e6 || reps >= (std::size_t(1) << 24))
                    {
                        break;
                    }
                    const double grow = dt > 0 ? (o.targetMs * 1e6 * 1.2) / dt : 8.0;
                    reps = std::max<std::size_t>(reps + 1, static_cast<std::size_t>(reps * std::min(grow, 8.0)));
                }

                std::vector<double> perRun;
                for (int s = 0; s < samples; ++s)
                {
                    const double t0 = nowNs();
                    for (std::size_t r = 0; r < reps; ++r)
                    {
                        a->run();
                    }
                    perRun.push_back((nowNs() - t0) / static_cast<double>(reps));
                }
                const double med = median(perRun);
                const double mn = *std::min_element(perRun.begin(), perRun.end());
                const double mflops = equivalentFlops(kind, n) / med * 1e3;

                std::fprintf(csv, "%s,%s,%s,%zu,%d,%.1f,%.1f,%.1f,%.1f,%.3g,%.6g,ok\n", a->name().c_str(),
                             kindName(kind), sz.category.c_str(), n, a->effectiveThreads(o.threads), med, mn,
                             mflops, planUs, err, scale);
                std::fflush(csv);
                a->teardown();
            }
            std::fprintf(stderr, "# done %s n=%zu\n", kindName(kind), n);
        }
    }

    if (csv != stdout)
    {
        std::fclose(csv);
    }
    return 0;
}
