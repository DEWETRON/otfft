// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html
//
// NOTE: Do not include this file directly!

void unique_ptr_deleter(OTFFT::ComplexFFT *raw_pointer);
void unique_ptr_deleter(OTFFT::RealFFT *raw_pointer);
void unique_ptr_deleter(OTFFT::RealDCT *raw_pointer);

namespace Factory
{
    OTFFT::ComplexFFT* createComplexFFT(int n = 0);
    OTFFT::RealFFT* createRealFFT(int n = 0);
    OTFFT::RealDCT* createDCT(int n = 0);
    OTFFT::ComplexFFT* createBluesteinFFT(int n = 0);
}
