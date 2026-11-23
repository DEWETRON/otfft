// Copyright (C) DEWETRON GmbH 2017
#pragma once

// Test framework (doctest, vendored in third_party/doctest) plus the few
// numeric checks the tests use.

#include <doctest/doctest.h>

#include <cmath>

namespace OTFFT
{
namespace test
{
    // True when a and b differ by at most pct percent of *each* of them
    // (the same "strong" comparison Boost.Test's BOOST_CHECK_CLOSE used).
    inline bool close_pct(double a, double b, double pct)
    {
        if (a == b)
        {
            return true;
        }
        const double diff = std::fabs(a - b);
        const double tol = pct / 100.0;
        return diff <= tol * std::fabs(a) && diff <= tol * std::fabs(b);
    }
}
}

// |x| <= tol
#define CHECK_SMALL(x, tol) CHECK_MESSAGE(std::fabs(x) <= (tol), (x) << " is not small (tol " << (tol) << ")")

// a and b are equal within pct percent
#define CHECK_CLOSE_PCT(a, b, pct) \
    CHECK_MESSAGE(OTFFT::test::close_pct((a), (b), (pct)), (a) << " is not close to " << (b) << " (" << (pct) << "%)")
