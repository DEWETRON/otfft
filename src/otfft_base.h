// Copyright (c) 2015, OK おじさん(岡久卓也)
// Copyright (c) 2015, OK Ojisan(Takuya OKAHISA)
// Copyright (c) 2017 to the present, DEWETRON GmbH
// OTFFT Implementation Version 9.5
// based on Stockham FFT algorithm
// from OK Ojisan(Takuya OKAHISA), source: http://www.moon.sannet.ne.jp/okahisa/stockham/stockham.html

using namespace OTFFT;
using namespace OTFFT_MISC;

/******************************************************************************
*  Complex FFT
******************************************************************************/

struct FFT0
{
    std::unique_ptr<FFT_IF> obj;
    int N, log_N;

    FFT0() noexcept;
    FFT0(int n);
    ~FFT0() noexcept;

    void setup(int n);

    void fwd(complex_vector  x, complex_vector y) const noexcept;
    void fwd0(complex_vector x, complex_vector y) const noexcept;
    void fwdu(complex_vector x, complex_vector y) const noexcept;
    void fwdn(complex_vector x, complex_vector y) const noexcept;
    void inv(complex_vector  x, complex_vector y) const noexcept;
    void inv0(complex_vector x, complex_vector y) const noexcept;
    void invu(complex_vector x, complex_vector y) const noexcept;
    void invn(complex_vector x, complex_vector y) const noexcept;
};

class FFT : public ComplexFFT
{
public:
    FFT() noexcept;
    FFT(int n);
    ~FFT();

    void setup(int n) override;

    void fwd(complex_vector  x) const noexcept override;
    void fwd0(complex_vector x) const noexcept override;
    void fwdu(complex_vector x) const noexcept override;
    void fwdn(complex_vector x) const noexcept override;
    void inv(complex_vector  x) const noexcept override;
    void inv0(complex_vector x) const noexcept override;
    void invu(complex_vector x) const noexcept override;
    void invn(complex_vector x) const noexcept override;

private:
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* y;
};

/******************************************************************************
*  Real FFT
******************************************************************************/

class RFFT : public RealFFT
{
public:
    RFFT() noexcept;
    RFFT(int n);
    ~RFFT();

    void setup(int n) override;

    void fwd(const_double_vector  x, complex_vector y) const noexcept override;
    void fwd0(const_double_vector x, complex_vector y) const noexcept override;
    void fwdu(const_double_vector x, complex_vector y) const noexcept override;
    void fwdn(const_double_vector x, complex_vector y) const noexcept override;
    void inv(complex_vector  x, double_vector y) const noexcept override;
    void inv0(complex_vector x, double_vector y) const noexcept override;
    void invu(complex_vector x, double_vector y) const noexcept override;
    void invn(complex_vector x, double_vector y) const noexcept override;

private:
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;

    int N;
    FFT0 fft;
    simd_array<complex_t> weight;
    complex_t* U;
};

/******************************************************************************
*  DCT
******************************************************************************/

struct DCT0
{
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;

    int N;
    RFFT rfft;
    simd_array<complex_t> weight;
    complex_t* V;

    DCT0() noexcept;
    DCT0(int n);
    ~DCT0();

    void setup(int n);

    void fwd(double_vector  x, double_vector y, complex_vector z) const noexcept;
    void fwd0(double_vector x, double_vector y, complex_vector z) const noexcept;
    void fwdn(double_vector x, double_vector y, complex_vector z) const noexcept;
    void inv(double_vector  x, double_vector y, complex_vector z) const noexcept;
    void inv0(double_vector x, double_vector y, complex_vector z) const noexcept;
    void invn(double_vector x, double_vector y, complex_vector z) const noexcept;
};

class DCT : public RealDCT
{
public:
    DCT() noexcept;
    DCT(int n);
    ~DCT();

    void setup(int n) override;

    void fwd(double_vector  x) const noexcept override;
    void fwd0(double_vector x) const noexcept override;
    void fwdn(double_vector x) const noexcept override;
    void inv(double_vector  x) const noexcept override;
    void inv0(double_vector x) const noexcept override;
    void invn(double_vector x) const noexcept override;

private:
    int N;
    DCT0 dct;
    simd_array<double> work1;
    simd_array<complex_t> work2;
    double* y;
    complex_t* z;
};

/******************************************************************************
*  Bluestein's FFT
******************************************************************************/

class Bluestein : public ComplexFFT
{
public:
    Bluestein() noexcept;
    Bluestein(int n);
    ~Bluestein();

    void setup(int n) override;

    void fwd(complex_vector  x) const noexcept override;
    void fwd0(complex_vector x) const noexcept override;
    void fwdu(complex_vector x) const noexcept override;
    void fwdn(complex_vector x) const noexcept override;
    void inv(complex_vector  x) const noexcept override;
    void inv0(complex_vector x) const noexcept override;
    void invu(complex_vector x) const noexcept override;
    void invn(complex_vector x) const noexcept override;

private:
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;

    int N, L;
    FFT fft;
    simd_array<complex_t> work1;
    simd_array<complex_t> work2;
    simd_array<complex_t> weight;
    complex_t* a;
    complex_t* b;
    complex_t* W;
};
