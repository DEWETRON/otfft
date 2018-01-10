// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

using namespace OTFFT;

typedef struct FFT_IF
{
    virtual ~FFT_IF() {}

    virtual void setup(int n) = 0;
    virtual void setup2(const int n) = 0;
    virtual void fwd(complex_vector x, complex_vector y) const noexcept = 0;
    virtual void fwd0(complex_vector x, complex_vector y) const noexcept = 0;
    virtual void fwdu(complex_vector x, complex_vector y) const noexcept = 0;
    virtual void fwdn(complex_vector x, complex_vector y) const noexcept = 0;
    virtual void inv(complex_vector x, complex_vector y) const noexcept = 0;
    virtual void inv0(complex_vector x, complex_vector y) const noexcept = 0;
    virtual void invu(complex_vector x, complex_vector y) const noexcept = 0;
    virtual void invn(complex_vector x, complex_vector y) const noexcept = 0;
} FFT_IF;
