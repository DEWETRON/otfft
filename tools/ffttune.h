// Shared declarations of the ffttune tool (see tools/ffttune.cpp).
#pragma once

namespace ffttune
{
    // The candidates are numbered like the upstream OTFFT tool and FFT1..FFT7 in
    // src/otfft_algo.cpp: DIF4, DIT4, DIF8, DIT8, DIF16, DIT16, SixStep (names in
    // ffttune.cpp); number 8 is TwoPhase (FFT_TWOPHASE, lengths 2^16 to 2^22 only, D35).
    // OTFFT_MixedRadix (FFT8 there) is the fallback for other lengths and is not a candidate.
    //
    // This header is also compiled with the library's flags (C++11 with GCC), so it
    // stays C++11; ffttune.cpp itself is C++17.
    constexpr int kAlgorithms = 8;

    // Times one forward plus one inverse transform of length 2^log_n for each candidate.
    // us[i] is the median time in microseconds, or a negative number when the candidate
    // gave a result that differs from DIF4 by more than 1e-9 (relative) or could not be
    // created. Returns false only on an allocation failure.
    typedef bool (*MeasureFn)(int log_n, int repeats, double batch_ms, double us[kAlgorithms]);
}
