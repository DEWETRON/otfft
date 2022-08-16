// Copyright DEWETRON GmbH 2017
// based on OK-Ojisan Template FFT

#include "otfft_types.h"
#include "otfft_misc.h"
#include "otfft.h"
#include "otfft_platform.h"

#ifdef USE_AVX2
#include "otfft_avx2.h"
#endif
#ifdef USE_AVX
#include "otfft_avx.h"
#endif
#ifdef USE_SSE2
#include "otfft_sse2.h"
#endif

#include <cassert>
#include <stdint.h>
#include <iostream>

namespace OTFFT_NAMESPACE
{
    namespace OTFFT_AVXDIF4 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT4 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF8 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT8 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_SixStep { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIF16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_AVXDIT16 { extern std::unique_ptr<FFT_IF> instance(); }
    namespace OTFFT_MixedRadix { extern std::unique_ptr<FFT_IF> instance(); }

    // see upstream otfft.cpp
    namespace FFT1 = OTFFT_AVXDIF4;
    namespace FFT2 = OTFFT_AVXDIT4;
    namespace FFT3 = OTFFT_AVXDIF8;
    namespace FFT4 = OTFFT_AVXDIT8;
    namespace FFT5 = OTFFT_AVXDIF16;
    namespace FFT6 = OTFFT_AVXDIT16;
    namespace FFT7 = OTFFT_SixStep;
    namespace FFT8 = OTFFT_MixedRadix;

    /******************************************************************************
    *  Complex FFT
    ******************************************************************************/

    FFT0::FFT0() noexcept : obj(), N(0), log_N(0) {}

    FFT0::FFT0(int n) : obj(), N(n), log_N(0)
    {
        setup(n);
    }

    FFT0::~FFT0() noexcept
    {
        obj.reset();
    }

    void FFT0::setup(int n)
    {
        obj.reset();
        for (N = n, log_N = 0; n > 1; n >>= 1) log_N++;
        if (N != (1 << log_N)) log_N = -1;
        try
        {
            // see upstream otfft_gen_setup.h
            switch (log_N)
            {
            case  0: break;
            case  1: obj = FFT2::instance(); obj->setup2(log_N); break;
            case  2: obj = FFT1::instance(); obj->setup2(log_N); break;
            case  3: obj = FFT4::instance(); obj->setup2(log_N); break;
            case  4: obj = FFT1::instance(); obj->setup2(log_N); break;
            case  5: obj = FFT4::instance(); obj->setup2(log_N); break;
            case  6: obj = FFT3::instance(); obj->setup2(log_N); break;
            case  7: obj = FFT5::instance(); obj->setup2(log_N); break;
            case  8: obj = FFT3::instance(); obj->setup2(log_N); break;
            case  9: obj = FFT3::instance(); obj->setup2(log_N); break;
            case 10: obj = FFT1::instance(); obj->setup2(log_N); break;
            case 11: obj = FFT5::instance(); obj->setup2(log_N); break;
            case 12: obj = FFT4::instance(); obj->setup2(log_N); break;
            case 13: obj = FFT3::instance(); obj->setup2(log_N); break;
            case 14: obj = FFT4::instance(); obj->setup2(log_N); break;
            case 15: obj = FFT4::instance(); obj->setup2(log_N); break;
            case 16: obj = FFT7::instance(); obj->setup2(log_N); break;
            case 17: obj = FFT4::instance(); obj->setup2(log_N); break;
            case 18: obj = FFT7::instance(); obj->setup2(log_N); break;
            case 19: obj = FFT7::instance(); obj->setup2(log_N); break;
            case 20: obj = FFT7::instance(); obj->setup2(log_N); break;
            case 21: obj = FFT7::instance(); obj->setup2(log_N); break;
            case 22: obj = FFT7::instance(); obj->setup2(log_N); break;
            case 23: obj = FFT7::instance(); obj->setup2(log_N); break;
            case 24: obj = FFT7::instance(); obj->setup2(log_N); break;
            default: obj = FFT8::instance(); obj->setup(N); break;
            }
        }
        catch (...)
        {
            obj.reset();
            throw;
        }
    }

    void FFT0::fwd(complex_vector x, complex_vector y) const noexcept
    {
        obj->fwd(x, y);
    }

    void FFT0::fwd0(complex_vector x, complex_vector y) const noexcept
    {
        obj->fwd0(x, y);
    }

    void FFT0::fwdu(complex_vector x, complex_vector y) const noexcept
    {
        obj->fwdu(x, y);
    }

    void FFT0::fwdn(complex_vector x, complex_vector y) const noexcept
    {
        fwd(x, y);
    }

    void FFT0::inv(complex_vector x, complex_vector y) const noexcept
    {
        obj->inv(x, y);
    }

    void FFT0::inv0(complex_vector x, complex_vector y) const noexcept
    {
        inv(x, y);
    }

    void FFT0::invu(complex_vector x, complex_vector y) const noexcept
    {
        obj->invu(x, y);
    }

    void FFT0::invn(complex_vector x, complex_vector y) const noexcept
    {
        obj->invn(x, y);
    }

    FFT::FFT() noexcept : fft(), work(), y(0) {}

    FFT::FFT(int n) : fft(n), work(n), y(&work) {}

    FFT::~FFT()
    {
        work.destroy();
    }

    void FFT::setup(int n)
    {
        fft.setup(n);
        work.setup(n);
        y = &work;
    }

    void FFT::fwd(complex_vector  x) const noexcept
    {
        fft.fwd(x, y);
    }

    void FFT::fwd0(complex_vector x) const noexcept
    {
        fft.fwd0(x, y);
    }

    void FFT::fwdu(complex_vector x) const noexcept
    {
        fft.fwdu(x, y);
    }

    void FFT::fwdn(complex_vector x) const noexcept
    {
        fft.fwdn(x, y);
    }

    void FFT::inv(complex_vector  x) const noexcept
    {
        fft.inv(x, y);
    }

    void FFT::inv0(complex_vector x) const noexcept
    {
        fft.inv0(x, y);
    }

    void FFT::invu(complex_vector x) const noexcept
    {
        fft.invu(x, y);
    }

    void FFT::invn(complex_vector x) const noexcept
    {
        fft.invn(x, y);
    }

    /******************************************************************************
    *  Real FFT
    ******************************************************************************/

    RFFT::RFFT() noexcept : N(0), fft(), weight(), U(0) {}

    RFFT::RFFT(int n) : N(0), fft(), weight(), U(0)
    {
        setup(n);
    }

    RFFT::~RFFT()
    {
        weight.destroy();
    }

    void RFFT::setup(int n)
    {
        assert(n <= 1 || (n & 1) == 0);
        int log_N;
        N = n;
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        fft.setup(N/2);
        weight.setup(N+1); U = &weight;
        const double theta0 = 2*M_PI/N;
        const int Nh = N/2;
        const int Nq = N/4;
        const int Ne = N/8;
        const int Nd = N - Nq;
        if (N < 1) {}
        else if (N != (1 << log_N)) for (int p = 0; p <= Nh; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            U[p]    = complex_t(1 - s,  c)/2;
            U[N-p]  = complex_t(1 + s,  c)/2;
        }
        else if (N == 1) { U[0] = U[1] = complex_t(1, 1)/2; }
        else if (N == 2) {
            U[0] = U[2] = complex_t(1,  1)/2;
            U[1]        = complex_t(1, -1)/2;
        }
        else if (N == 4) {
            U[0] = complex_t(1 + 0,  1)/2;
            U[1] = complex_t(1 + 1,  0)/2;
            U[2] = complex_t(1 + 0, -1)/2;
            U[3] = complex_t(1 - 1,  0)/2;
            U[4] = complex_t(1 + 0,  1)/2;
        }
        else if (N < OMP_THRESHOLD_W) for (int p = 0; p <= Ne; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            U[p]    = complex_t(1 - s,  c)/2;
            U[Nq-p] = complex_t(1 + c, -s)/2;
            U[Nq+p] = complex_t(1 + c,  s)/2;
            U[Nh-p] = complex_t(1 - s, -c)/2;
            U[Nh+p] = complex_t(1 + s, -c)/2;
            U[Nd-p] = complex_t(1 - c,  s)/2;
            U[Nd+p] = complex_t(1 - c, -s)/2;
            U[N-p]  = complex_t(1 + s,  c)/2;
        }
        else
        #pragma omp parallel for schedule(static)
            for (int p = 0; p <= Ne; p++) {
                const double theta = p * theta0;
                const double c =  cos(theta);
                const double s = -sin(theta);
                U[p]    = complex_t(1 - s,  c)/2;
                U[Nq-p] = complex_t(1 + c, -s)/2;
                U[Nq+p] = complex_t(1 + c,  s)/2;
                U[Nh-p] = complex_t(1 - s, -c)/2;
                U[Nh+p] = complex_t(1 + s, -c)/2;
                U[Nd-p] = complex_t(1 - c,  s)/2;
                U[Nd+p] = complex_t(1 - c, -s)/2;
                U[N-p]  = complex_t(1 + s,  c)/2;
            }
    }

    void RFFT::fwd(const_double_vector x, complex_vector y) const noexcept
    {
        if (N < 1) return;
        else if (N == 1) { y[0] = x[0]; return; }
        const xmm rN = cmplx(1.0/N, 1.0/N);
        const int Nh = N/2;
        const int Nq = N/4;
        complex_vector z = y + Nh;
        for (int p = 0; p < Nh; p++) setpz(z[p], getpz(x + 2*p));
        fft.fwd0(z, y);
        y[0] = (z[0].Re + z[0].Im) / N;
        z[0] = (z[0].Re - z[0].Im) / N;
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],    mulpd(rN,       subpz(a, c)));
                setpz(y[Nh-k], mulpd(rN, cnjpz(addpz(b, c))));
            }
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],    mulpd(rN,       subpz(a, c)));
                setpz(y[Nh-k], mulpd(rN, cnjpz(addpz(b, c))));
            }
            #pragma omp for schedule(static)
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
    }

    void RFFT::fwd0(const_double_vector x, complex_vector y) const noexcept
    {
        if (N < 1) return;
        else if (N == 1) { y[0] = x[0]; return; }
        const int Nh = N/2;
        const int Nq = N/4;
        complex_vector z = y + Nh;
        for (int p = 0; p < Nh; p++) setpz(z[p], getpz(x + 2*p));
        fft.fwd0(z, y);
        y[0] = z[0].Re + z[0].Im;
        z[0] = z[0].Re - z[0].Im;
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],          subpz(a, c));
                setpz(y[Nh-k], cnjpz(addpz(b, c)));
            }
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],          subpz(a, c));
                setpz(y[Nh-k], cnjpz(addpz(b, c)));
            }
            #pragma omp for schedule(static)
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
    }

    void RFFT::fwdu(const_double_vector x, complex_vector y) const noexcept
    {
        if (N < 1) return;
        else if (N == 1) { y[0] = x[0]; return; }
        const double sN = sqrt(double(N));
        const xmm rsN = cmplx(1.0/sN, 1.0/sN);
        const int Nh = N/2;
        const int Nq = N/4;
        complex_vector z = y + Nh;
        for (int p = 0; p < Nh; p++) setpz(z[p], getpz(x + 2*p));
        fft.fwd0(z, y);
        y[0] = (z[0].Re + z[0].Im) / sN;
        z[0] = (z[0].Re - z[0].Im) / sN;
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],    mulpd(rsN,       subpz(a, c)));
                setpz(y[Nh-k], mulpd(rsN, cnjpz(addpz(b, c))));
            }
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],    mulpd(rsN,       subpz(a, c)));
                setpz(y[Nh-k], mulpd(rsN, cnjpz(addpz(b, c))));
            }
            #pragma omp for schedule(static)
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
    }

    void RFFT::fwdn(const_double_vector x, complex_vector y) const noexcept
    {
        fwd(x, y);
    }

    void RFFT::inv(complex_vector x, double_vector y) const noexcept
    {
        if (N < 1) return;
        else if (N == 1) { y[0] = x[0].Re; return; }
        static const xmm x2 = { 2.0, 2.0 };
        const int Nh = N/2;
        complex_vector z = x + Nh;
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], mulpd(x2, cnjpz(subpz(a, c))));
            }
            fft.inv0(z, x);
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], mulpd(x2, cnjpz(subpz(a, c))));
            }
            fft.inv0(z, x);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
    }

    void RFFT::inv0(complex_vector x, double_vector y) const noexcept
    {
        inv(x, y);
    }

    void RFFT::invu(complex_vector x, double_vector y) const noexcept
    {
        if (N < 1) return;
        else if (N == 1) { y[0] = x[0].Re; return; }
        const double s2dsN = 2.0/sqrt(N);
        const xmm x2dsN = cmplx(s2dsN, s2dsN);
        const int Nh = N/2;
        complex_vector z = x + Nh;
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], mulpd(x2dsN, cnjpz(subpz(a, c))));
            }
            fft.inv0(z, x);
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], mulpd(x2dsN, cnjpz(subpz(a, c))));
            }
            fft.inv0(z, x);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
    }

    void RFFT::invn(complex_vector x, double_vector y) const noexcept
    {
        if (N < 1) return;
        else if (N == 1) { y[0] = x[0].Re; return; }
        const int Nh = N/2;
        complex_vector z = x + Nh;
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], cnjpz(subpz(a, c)));
            }
            fft.invn(z, x);
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], cnjpz(subpz(a, c)));
            }
            fft.invn(z, x);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
    }

    /******************************************************************************
    *  DCT-II
    ******************************************************************************/

    DCT0::DCT0() noexcept : N(0), rfft(), weight(), V(0) {}

    DCT0::DCT0(int n) : N(0), rfft(), weight(), V(0)
    {
        setup(n);
    }

    DCT0::~DCT0()
    {
        weight.destroy();
    }

    void DCT0::setup(int n)
    {
        assert(n <= 1 || (n & 1) == 0);
        N = n;
        rfft.setup(N);
        weight.setup(N+1); V = &weight;
        const double theta0 = M_PI/(2*N);
        const int Nh = N/2;
        if (N < 2) {}
        else if (N < OMP_THRESHOLD_W) for (int p = 0; p <= Nh; p++) {
            const double theta = p * theta0;
            const double c = cos(theta);
            const double s = sin(theta);
            V[p]    = complex_t(c, s);
            V[N-p]  = complex_t(s, c);
        }
        else
        #pragma omp parallel for schedule(static)
        for (int p = 0; p <= Nh; p++) {
            const double theta = p * theta0;
            const double c = cos(theta);
            const double s = sin(theta);
            V[p]    = complex_t(c, s);
            V[N-p]  = complex_t(s, c);
        }
    }

    void DCT0::fwd(double_vector x, double_vector y, complex_vector z) const noexcept
    {
        if (N < 2) return;
        const int Nh = N/2;
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        rfft.fwd(y, z);
        if (N < OMP_THRESHOLD) {
            //for (int k = 0; k < N; k++)
            //    x[k] = V[k].Re*z[k].Re + V[k].Im*z[k].Im;
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
    }

    void DCT0::fwd0(double_vector x, double_vector y, complex_vector z) const noexcept
    {
        if (N < 2) return;
        const int Nh = N/2;
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        rfft.fwd0(y, z);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
    }

    void DCT0::fwdn(double_vector x, double_vector y, complex_vector z) const noexcept
    {
        fwd(x, y, z);
    }

    void DCT0::inv(double_vector x, double_vector y, complex_vector z) const noexcept
    {
        if (N < 2) return;
        const int Nh = N/2;
        z[0] = x[0];
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        rfft.inv(z, y);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
    }

    void DCT0::inv0(double_vector x, double_vector y, complex_vector z) const noexcept
    {
        inv(x, y, z);
    }

    void DCT0::invn(double_vector x, double_vector y, complex_vector z) const noexcept
    {
        if (N < 2) return;
        const int Nh = N/2;
        z[0] = x[0];
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        rfft.invn(z, y);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
    }

    DCT::DCT() noexcept : N(0), dct(), work1(), work2(), y(0), z(0) {}

    DCT::DCT(int n) : N(0), dct(), work1(), work2(), y(0), z(0)
    {
        setup(n);
    }

    DCT::~DCT()
    {
        work1.destroy();
        work2.destroy();
    }

    void DCT::setup(int n)
    {
        N = n;
        dct.setup(N);
        work1.setup(N); y = &work1;
        work2.setup(N); z = &work2;
    }

    void DCT::fwd(double_vector  x) const noexcept
    {
        dct.fwd(x, y, z);
    }

    void DCT::fwd0(double_vector x) const noexcept
    {
        dct.fwd0(x, y, z);
    }

    void DCT::fwdn(double_vector x) const noexcept
    {
        dct.fwdn(x, y, z);
    }

    void DCT::inv(double_vector  x) const noexcept
    {
        dct.inv(x, y, z);
    }

    void DCT::inv0(double_vector x) const noexcept
    {
        dct.inv0(x, y, z);
    }

    void DCT::invn(double_vector x) const noexcept
    {
        dct.invn(x, y, z);
    }

    /******************************************************************************
    *  Bluestein's FFT
    ******************************************************************************/

    Bluestein::Bluestein() noexcept : N(0), L(0), fft(), work1(), work2(), weight(), a(0), b(0), W(0) {}

    Bluestein::Bluestein(int n) : N(0), L(0), fft(), work1(), work2(), weight(), a(0), b(0), W(0)
    {
        setup(n);
    }

    Bluestein::~Bluestein()
    {
        work1.destroy();
        work2.destroy();
        weight.destroy();
    }

    void Bluestein::setup(int n)
    {
        if (n < 1) return;
        N = n;
        const int N2 = 2*N;
        for (L = 1; L < N2 - 1; L *= 2);
        fft.setup(L);
        work1.setup(L); a = &work1;
        work2.setup(L); b = &work2;
        weight.setup(N2+1); W = &weight;
        const double theta0 = M_PI/N;
        W[0] = W[N2] = 1; W[N] = -1;
        if (N < OMP_THRESHOLD_W) for (int p = 1; p < N; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            W[p]    = complex_t(c,  s);
            W[N2-p] = complex_t(c, -s);
        }
        else
        #pragma omp parallel for schedule(static)
        for (int p = 1; p < N; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            W[p]    = complex_t(c,  s);
            W[N2-p] = complex_t(c, -s);
        }
    }

    void Bluestein::fwd(complex_vector x) const noexcept
    {
        if (N < 2) return;
        const xmm rN = cmplx(1.0/N, 1.0/N);
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            #pragma omp for schedule(static)
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
    }

    void Bluestein::fwd0(complex_vector x) const noexcept
    {
        if (N < 2) return;
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            #pragma omp for schedule(static)
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
    }

    void Bluestein::fwdu(complex_vector x) const noexcept
    {
        if (N < 2) return;
        const double ssrN = sqrt(1.0/N);
        const xmm srN = cmplx(ssrN, ssrN);
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            #pragma omp for schedule(static)
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(srN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(srN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
    }

    void Bluestein::fwdn(complex_vector x) const noexcept { fwd(x); }

    void Bluestein::inv(complex_vector x) const noexcept
    {
        if (N < 2) return;
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            #pragma omp for schedule(static)
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
    }

    void Bluestein::inv0(complex_vector x) const noexcept { inv(x); }

    void Bluestein::invu(complex_vector x) const noexcept
    {
        if (N < 2) return;
        const double ssrN = sqrt(1.0/N);
        const xmm srN = cmplx(ssrN, ssrN);
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            #pragma omp for schedule(static)
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(srN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(srN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
    }

    void Bluestein::invn(complex_vector x) const noexcept
    {
        if (N < 2) return;
        const xmm rN = cmplx(1.0/N, 1.0/N);
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            #pragma omp for schedule(static)
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
    }

    void unique_ptr_deleter(ComplexFFT *raw_pointer)
    {
        delete raw_pointer;
    }

    void unique_ptr_deleter(RealFFT *raw_pointer)
    {
        delete raw_pointer;
    }

    void unique_ptr_deleter(RealDCT *raw_pointer)
    {
        delete raw_pointer;
    }

    namespace Factory
    {
        ComplexFFT* createComplexFFT(int n)
        {
            return new FFT(n);
        }

        RealFFT* createRealFFT(int n)
        {
            return new RFFT(n);
        }

        RealDCT* createDCT(int n)
        {
            return new DCT(n);
        }

        ComplexFFT* createBluesteinFFT(int n)
        {
            return new Bluestein(n);
        }
    }
}
