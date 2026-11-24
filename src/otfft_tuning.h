// Run-time overrides of the per-length algorithm tables (roadmap R1.8, decision D22).
// Internal: the public side is OTFFT::loadTuningFile and OTFFT::resetTuning in otfft.h.
#ifndef OTFFT_TUNING_H
#define OTFFT_TUNING_H

namespace OTFFT
{
    namespace Tuning
    {
        enum Backend
        {
            BACKEND_SSE2 = 0,
            BACKEND_AVX = 1,
            BACKEND_AVX2 = 2,
            BACKEND_COUNT = 3
        };

        // Algorithm number (1 to 7, as in otfft_algo.cpp) loaded for this backend, table and
        // length 2^log_N, or 0 if none: then the built-in table applies. Thread safe.
        int lookup(Backend backend, bool threaded, int log_N) noexcept;
    }
}

#endif
