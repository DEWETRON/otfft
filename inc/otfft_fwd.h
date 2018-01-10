// Copyright (C) DEWETRON GmbH 2017

#pragma once

#include <memory>

namespace OTFFT
{
    struct complex_t;

    class ComplexFFT;
    class RealFFT;
    class RealDCT;

    class unique_ptr_delete;

    using ComplexFFTPtr = std::unique_ptr<ComplexFFT, unique_ptr_delete>;
    using RealFFTPtr = std::unique_ptr<RealFFT, unique_ptr_delete>;
    using RealDCTPtr = std::unique_ptr<RealDCT, unique_ptr_delete>;
}
