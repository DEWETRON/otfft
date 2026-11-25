/******************************************************************************
*  OTFFT Header Version 11.4xv
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_h
#define otfft_h

#include "otfft_config.h"
#include "otfft_fwd.h"
#include "otfft_types.h"

#include <memory>
#include <string>
#include <cstdlib>
#include <cstddef>
#include <cstdint>

namespace OTFFT
{
    enum class TransformationType
    {
        // Complex Discrete Fourier Transform
        TRANSFORM_FFT_COMPLEX,
        // Real Discrete Fourier Transform
        TRANSFORM_FFT_REAL,
        // Discrete Cosine Transform (DCT-II)
        TRANSFORM_DCT,
        // Bluestein's FFT
        TRANSFORM_BLUESTEIN
    };

    enum class OptimizationType
    {
        OPTIMIZED_FFT_AUTO,
#ifdef OTFFT_WITH_SSE2
        OPTIMIZED_FFT_SSE2,
#endif
#ifdef OTFFT_WITH_AVX
        OPTIMIZED_FFT_AVX,
#endif
#ifdef OTFFT_WITH_AVX2
        OPTIMIZED_FFT_AVX2,
#endif
#ifdef OTFFT_WITH_AVX512
        /// Only used when asked for, never chosen by OPTIMIZED_FFT_AUTO (decision D32)
        OPTIMIZED_FFT_AVX512
#endif
    };

    /**
     * @brief Complex In-Place Discrete Fourier Transform
     */
    class ComplexFFT
    {
    public:
        virtual ~ComplexFFT() = default;

        /**
         * @brief Setup the sequence length of the algorithm (up to 2^30)
         */
        virtual void setup(int n) = 0;

        /**
         * @brief Discrete forward transformation (with normalization by 1/N)
         */
        virtual void fwd(complex_vector  x) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (without normalization)
         */
        virtual void fwd0(complex_vector x) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (unitary transformation, with normalization by sqrt(1/N))
         */
        virtual void fwdu(complex_vector x) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (with normalization by 1/N)
         */
        virtual void fwdn(complex_vector x) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (without normalization)
         */
        virtual void inv(complex_vector  x) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (without normalization)
         */
        virtual void inv0(complex_vector x) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (unitary transformation)
         */
        virtual void invu(complex_vector x) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (with normalization by 1/N)
         */
        virtual void invn(complex_vector x) const noexcept = 0;
    };

    /**
     * @brief Real Out-Of-Place Discrete Fourier Transform
     */
    class RealFFT
    {
    public:
        virtual ~RealFFT() = default;

        /**
         * @brief Setup the sequence length of the algorithm (up to 2^30)
         */
        virtual void setup(int n) = 0;

        /**
         * @brief Discrete forward transformation (with normalization by 1/N)
         */
        virtual void fwd(const_double_vector  x, complex_vector y) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (without normalization)
         */
        virtual void fwd0(const_double_vector x, complex_vector y) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (unitary transformation, with normalization by sqrt(1/N))
         */
        virtual void fwdu(const_double_vector x, complex_vector y) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (with normalization by 1/N)
         */
        virtual void fwdn(const_double_vector x, complex_vector y) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (without normalization)
         */
        virtual void inv(complex_vector  x, double_vector y) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (without normalization)
         */
        virtual void inv0(complex_vector x, double_vector y) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (unitary transformation)
         */
        virtual void invu(complex_vector x, double_vector y) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (with normalization by 1/N)
         */
        virtual void invn(complex_vector x, double_vector y) const noexcept = 0;
    };

    /**
     * @brief Real In-Place Discrete Cosine Transform (DCT-II)
     */
    class RealDCT
    {
    public:
        virtual ~RealDCT() = default;

        /**
         * @brief Setup the sequence length of the algorithm (up to 2^30)
         */
        virtual void setup(int n) = 0;

        /**
         * @brief Discrete forward transformation (with normalization by 1/N)
         */
        virtual void fwd(double_vector  x) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (without normalization)
         */
        virtual void fwd0(double_vector x) const noexcept = 0;
        /**
         * @brief Discrete forward transformation (with normalization by 1/N)
         */
        virtual void fwdn(double_vector x) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (without normalization)
         */
        virtual void inv(double_vector  x) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (without normalization)
         */
        virtual void inv0(double_vector x) const noexcept = 0;
        /**
         * @brief Discrete inverse transformation (with normalization by 1/N)
         */
        virtual void invn(double_vector x) const noexcept = 0;
    };

    // The following are the custom delete functors used by unique_ptr
    // for the FFT classes.
    class unique_ptr_delete
    {
    public:
        void operator()(ComplexFFT *raw_pointer);
        void operator()(RealFFT *raw_pointer);
        void operator()(RealDCT *raw_pointer);
    };

    /**
     * @brief Create a transformation instance with an optional sequence length
     *
     * The sequence length has to be an even number (up to 2^30) if transformation type is one of:
     *      - ComplexFFT
     *      - RealFFT
     *      - DCT
     *
     * NOTE: The Bluestein's ComplexFFT supports sequence length of any natural number of 2^28 or less.
     */
    /**
     * @brief Replace the built-in per-length algorithm choice of the complex FFT by a tuning file.
     *
     * The file lists, for a backend (sse2, avx, avx2, avx512), a table (serial or threaded) and a length
     * 2^N, the algorithm to use. Lengths it does not list keep the built-in choice. The file
     * replaces any file loaded before. If it cannot be read or has an error nothing changes,
     * false is returned and `error` says why. Only transform objects created or set up after
     * the call use it. Call it before creating transforms; it is safe to call while other
     * threads create them, but they then see either the old or the new table.
     */
    bool loadTuningFile(const std::string& path, std::string& error);

    /**
     * @brief Drop a loaded tuning file: the built-in algorithm choice applies again.
     */
    void resetTuning();

    namespace Factory
    {
        ComplexFFTPtr createComplexFFT(int n = 0, OptimizationType t = OptimizationType::OPTIMIZED_FFT_AUTO);
        RealFFTPtr createRealFFT(int n = 0, OptimizationType t = OptimizationType::OPTIMIZED_FFT_AUTO);
        RealDCTPtr createDCT(int n = 0, OptimizationType t = OptimizationType::OPTIMIZED_FFT_AUTO);
        ComplexFFTPtr createBluesteinFFT(int n = 0, OptimizationType t = OptimizationType::OPTIMIZED_FFT_AUTO);
    }
}

#endif // otfft_h
