/******************************************************************************
*  OTFFT Header Version 11.4xv
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#pragma once

//=============================================================================
// Customizing Parameter
//=============================================================================

#define USE_UNALIGNED_MEMORY 1

//=============================================================================

#include <memory>
#include <cmath>

#include "otfft_complex.h"

//=============================================================================
// noexcept is not supported for all compilers
#if defined(__clang__)
#  if !__has_feature(cxx_noexcept)
#    define noexcept
#  endif
#else
#  if defined(__GXX_EXPERIMENTAL_CXX0X__) && __GNUC__ * 10 + __GNUC_MINOR__ >= 46 || \
      defined(_MSC_VER) && _MSC_VER >= 1900
	// everything fine ...
#  else
#    define noexcept
#  endif
#endif
//=============================================================================
