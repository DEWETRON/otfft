/******************************************************************************
*  OTFFT SixStep Version 11.4xv
*
*  Copyright (c) 2019 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_sixstep_h
#define otfft_sixstep_h

#include "otfft_avxdif16.h"

namespace OTFFT_NAMESPACE {

namespace OTFFT_SixStep { /////////////////////////////////////////////////////

using namespace OTFFT;
using namespace OTFFT_MISC;

typedef const_complex_vector weight_t;
struct index_t { short col, row; };
typedef const index_t* __restrict const const_index_vector;

} /////////////////////////////////////////////////////////////////////////////

}

#include "otfft_sixstepsq.h"
#include "otfft_eightstep.h"

#endif // otfft_sixstep_h
