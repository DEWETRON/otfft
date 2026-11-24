// Run-time overrides of the per-length algorithm tables (roadmap R1.8, decision D22).
// File format: docs/spec/factory.md (FAC-9 to FAC-13).
#include "otfft.h"
#include "otfft_tuning.h"

#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace OTFFT
{
    namespace
    {
        const int MAX_LOG = 24;

        struct Store
        {
            std::mutex lock;
            // [backend][0 serial, 1 threaded][log_N]; 0 = no override
            int table[Tuning::BACKEND_COUNT][2][MAX_LOG + 1];

            Store() { clear(); }

            void clear()
            {
                for (auto& b : table)
                    for (auto& t : b)
                        for (int& v : t) v = 0;
            }
        };

        Store& store()
        {
            static Store instance;
            return instance;
        }

        bool parseBackend(const std::string& s, int& out)
        {
            if (s == "sse2") out = Tuning::BACKEND_SSE2;
            else if (s == "avx") out = Tuning::BACKEND_AVX;
            else if (s == "avx2") out = Tuning::BACKEND_AVX2;
            else return false;
            return true;
        }

        bool parseAlgorithm(const std::string& s, int& out)
        {
            static const char* const names[] = {"DIF4", "DIT4", "DIF8", "DIT8", "DIF16", "DIT16", "SixStep"};
            for (int i = 0; i < 7; ++i)
            {
                if (s == names[i])
                {
                    out = i + 1;
                    return true;
                }
            }
            return false;
        }

        bool parseLog(const std::string& s, int& out)
        {
            if (s.empty() || s.size() > 2) return false;
            int v = 0;
            for (char c : s)
            {
                if (c < '0' || c > '9') return false;
                v = v * 10 + (c - '0');
            }
            if (v < 1 || v > MAX_LOG) return false;
            out = v;
            return true;
        }

        std::string lineError(int line, const std::string& what)
        {
            std::ostringstream os;
            os << "line " << line << ": " << what;
            return os.str();
        }
    }

    namespace Tuning
    {
        int lookup(Backend backend, bool threaded, int log_N) noexcept
        {
            if (log_N < 1 || log_N > MAX_LOG) return 0;
            Store& s = store();
            std::lock_guard<std::mutex> guard(s.lock);
            return s.table[backend][threaded ? 1 : 0][log_N];
        }
    }

    bool loadTuningFile(const std::string& path, std::string& error)
    {
        error.clear();
        std::ifstream in(path.c_str());
        if (!in)
        {
            error = "cannot open " + path;
            return false;
        }

        // parse into a scratch table; the loaded one changes only if the whole file is valid
        int scratch[Tuning::BACKEND_COUNT][2][MAX_LOG + 1] = {};
        bool haveHeader = false;
        std::string text;
        for (int line = 1; std::getline(in, text); ++line)
        {
            const std::string::size_type hash = text.find('#');
            if (hash != std::string::npos) text.erase(hash);
            std::istringstream words(text);
            std::string w[5];
            int count = 0;
            while (count < 5 && (words >> w[count])) ++count;
            if (count == 0) continue;
            if (!haveHeader)
            {
                if (count != 2 || w[0] != "otfft-tuning" || w[1] != "1")
                {
                    error = lineError(line, "expected the header 'otfft-tuning 1'");
                    return false;
                }
                haveHeader = true;
                continue;
            }
            if (count != 4)
            {
                error = lineError(line, "expected '<backend> <serial|threaded> <log2 N> <algorithm>'");
                return false;
            }
            int backend = 0;
            int log_N = 0;
            int algorithm = 0;
            if (!parseBackend(w[0], backend))
            {
                error = lineError(line, "unknown backend '" + w[0] + "' (sse2, avx or avx2)");
                return false;
            }
            if (w[1] != "serial" && w[1] != "threaded")
            {
                error = lineError(line, "unknown table '" + w[1] + "' (serial or threaded)");
                return false;
            }
            if (!parseLog(w[2], log_N))
            {
                error = lineError(line, "log2 N '" + w[2] + "' is not a number from 1 to 24");
                return false;
            }
            if (!parseAlgorithm(w[3], algorithm))
            {
                error = lineError(line, "unknown algorithm '" + w[3] + "' (DIF4, DIT4, DIF8, DIT8, DIF16, DIT16, SixStep)");
                return false;
            }
            scratch[backend][w[1] == "threaded" ? 1 : 0][log_N] = algorithm;
        }
        if (!haveHeader)
        {
            error = "the file is empty: expected the header 'otfft-tuning 1'";
            return false;
        }

        Store& s = store();
        std::lock_guard<std::mutex> guard(s.lock);
        for (int b = 0; b < Tuning::BACKEND_COUNT; ++b)
            for (int t = 0; t < 2; ++t)
                for (int n = 0; n <= MAX_LOG; ++n) s.table[b][t][n] = scratch[b][t][n];
        return true;
    }

    void resetTuning()
    {
        Store& s = store();
        std::lock_guard<std::mutex> guard(s.lock);
        s.clear();
    }
}
