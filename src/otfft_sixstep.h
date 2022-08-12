// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

#include "otfft_avxdif16.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_SixStep { /////////////////////////////////////////////////////

    using namespace OTFFT;
    using namespace OTFFT_MISC;

    typedef const_complex_vector weight_t;
    struct index_t { int row, col; };
    typedef const index_t* __restrict const const_index_vector;

} /////////////////////////////////////////////////////////////////////////////

}

#include "otfft_sixstepsq.h"
#include "otfft_eightstep.h"
