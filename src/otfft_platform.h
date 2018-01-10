// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

#pragma once

#if defined __APPLE__ && defined __MACH__
#define PLAT_MACOS
#endif

#if defined __linux__ || defined __gnu_linux__
#define PLAT_LINUX
#endif

#if defined _WIN32 \
    || defined __WIN32__ \
    || defined _WIN64 \
    || defined __WIN64__ \
    || defined __TOS_WIN__ \
    || defined __WINDOWS__
#define PLAT_WINDOWS
#endif

#if defined UNI_PLAT_WINDOWS && defined __GNUC__
#define PLAT_MINGW
#endif

// sanity check platform checks
#if ( defined PLAT_MACOS && defined PLAT_LINUX ) || ( defined PLAT_MACOS && defined PLAT_WINDOWS ) || ( defined PLAT_LINUX && defined PLAT_WINDOWS )
#error "unable to detect target platform"
#endif
