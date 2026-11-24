// ffttune: measure which algorithm is fastest for each power-of-two length on this
// machine and print the lines for the selection switch in src/otfft_algo.cpp.
//
// Port of the idea of upstream OTFFT's ffttune (otfft-11.6e/otfft/ffttune.cpp). Unlike
// upstream it does not generate files that the library includes: it only prints a
// table and, with --out, writes the suggested switch lines to a file. Applying them
// to src/otfft_algo.cpp is a separate, reviewed change.
//
// usage: ffttune [--backend sse2|avx|avx2] [--min-log N] [--max-log N] [--threads T]
//                [--repeats R] [--batch-ms MS] [--out FILE]
#include "ffttune.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FFTTUNE_HAVE_SSE2
extern bool ffttune_measure_sse2(int, int, double, double[ffttune::kAlgorithms]);
#endif
#ifdef FFTTUNE_HAVE_AVX
extern bool ffttune_measure_avx(int, int, double, double[ffttune::kAlgorithms]);
#endif
#ifdef FFTTUNE_HAVE_AVX2
extern bool ffttune_measure_avx2(int, int, double, double[ffttune::kAlgorithms]);
#endif

namespace
{
    constexpr std::array<const char*, ffttune::kAlgorithms> kNames = {"DIF4",  "DIT4",  "DIF8",   "DIT8",
                                                                     "DIF16", "DIT16", "SixStep"};

    struct Options
    {
        std::string backend;
        int minLog = 1;
        int maxLog = 24;
        int threads = 1;
        int repeats = 9;
        double batchMs = 20.0;
        std::string outPath;
    };

    ffttune::MeasureFn backendFunction(const std::string& name)
    {
#ifdef FFTTUNE_HAVE_SSE2
        if (name == "sse2") return ffttune_measure_sse2;
#endif
#ifdef FFTTUNE_HAVE_AVX
        if (name == "avx") return ffttune_measure_avx;
#endif
#ifdef FFTTUNE_HAVE_AVX2
        if (name == "avx2") return ffttune_measure_avx2;
#endif
        (void)name;
        return nullptr;
    }

    std::string defaultBackend()
    {
#ifdef FFTTUNE_HAVE_AVX2
        return "avx2";
#elif defined(FFTTUNE_HAVE_AVX)
        return "avx";
#else
        return "sse2";
#endif
    }

    std::string availableBackends()
    {
        std::string s;
#ifdef FFTTUNE_HAVE_SSE2
        s += " sse2";
#endif
#ifdef FFTTUNE_HAVE_AVX
        s += " avx";
#endif
#ifdef FFTTUNE_HAVE_AVX2
        s += " avx2";
#endif
        return s;
    }

    int usage()
    {
        std::fprintf(stderr,
                     "usage: ffttune [--backend%s] [--min-log N] [--max-log N]\n"
                     "               [--threads T] [--repeats R] [--batch-ms MS] [--out FILE]\n",
                     availableBackends().c_str());
        return 1;
    }

    bool parse(int argc, char** argv, Options& o)
    {
        o.backend = defaultBackend();
        for (int i = 1; i < argc; i++)
        {
            const std::string a = argv[i];
            if (i + 1 >= argc)
            {
                return false;
            }
            const std::string v = argv[++i];
            if (a == "--backend") o.backend = v;
            else if (a == "--min-log") o.minLog = std::atoi(v.c_str());
            else if (a == "--max-log") o.maxLog = std::atoi(v.c_str());
            else if (a == "--threads") o.threads = std::atoi(v.c_str());
            else if (a == "--repeats") o.repeats = std::atoi(v.c_str());
            else if (a == "--batch-ms") o.batchMs = std::atof(v.c_str());
            else if (a == "--out") o.outPath = v;
            else return false;
        }
        return o.minLog >= 1 && o.maxLog <= 24 && o.minLog <= o.maxLog && o.threads >= 1 && o.repeats >= 3 &&
               o.batchMs > 0;
    }
}

int main(int argc, char** argv)
{
    Options o;
    if (!parse(argc, argv, o))
    {
        return usage();
    }
    const ffttune::MeasureFn measure = backendFunction(o.backend);
    if (!measure)
    {
        std::fprintf(stderr, "ffttune: backend '%s' is not built (available:%s)\n", o.backend.c_str(),
                     availableBackends().c_str());
        return 1;
    }
#ifdef _OPENMP
    omp_set_num_threads(o.threads);
#endif

    std::printf("# ffttune: backend %s, %d thread(s), median of %d batches of about %.0f ms\n", o.backend.c_str(),
                o.threads, o.repeats, o.batchMs);
    std::printf("# time in microseconds of one forward plus one inverse transform; '-' = wrong result\n");
    std::printf("%6s", "log2N");
    for (int a = 0; a < ffttune::kAlgorithms; a++)
    {
        std::printf(" %10s", kNames[a]);
    }
    std::printf("  fastest\n");

    std::vector<std::string> lines;
    for (int n = o.minLog; n <= o.maxLog; n++)
    {
        std::array<double, ffttune::kAlgorithms> us;
        if (!measure(n, o.repeats, o.batchMs, us.data()))
        {
            std::fprintf(stderr, "ffttune: out of memory at 2^%d\n", n);
            return 1;
        }
        int best = -1;
        for (int a = 0; a < ffttune::kAlgorithms; a++)
        {
            if (us[a] >= 0 && (best < 0 || us[a] < us[best])) best = a;
        }
        if (best < 0)
        {
            std::fprintf(stderr, "ffttune: no usable algorithm at 2^%d\n", n);
            return 1;
        }
        // runner-up within 5%: the choice is not reliable
        bool tie = false;
        for (int a = 0; a < ffttune::kAlgorithms; a++)
        {
            if (a != best && us[a] >= 0 && us[a] < us[best] * 1.05) tie = true;
        }

        std::printf("%6d", n);
        for (int a = 0; a < ffttune::kAlgorithms; a++)
        {
            if (us[a] < 0) std::printf(" %10s", "-");
            else std::printf(" %10.*f", us[a] < 100 ? 3 : 0, us[a]);
        }
        std::printf("  %d (%s)%s\n", best + 1, kNames[best], tie ? " tie within 5%" : "");
        std::fflush(stdout);

        char line[96];
        std::snprintf(line, sizeof line, "case %2d: obj = FFT%d::instance(); obj->setup2(log_N); break;", n, best + 1);
        lines.emplace_back(line);
    }

    std::printf("\n# suggested lines for the switch in FFT0::setup (src/otfft_algo.cpp):\n");
    for (const std::string& l : lines)
    {
        std::printf("%s\n", l.c_str());
    }
    if (!o.outPath.empty())
    {
        std::ofstream ofs(o.outPath);
        if (!ofs)
        {
            std::fprintf(stderr, "ffttune: cannot write %s\n", o.outPath.c_str());
            return 1;
        }
        for (const std::string& l : lines)
        {
            ofs << l << "\n";
        }
    }
    return 0;
}
