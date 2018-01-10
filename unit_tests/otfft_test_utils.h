// Copyright (C) DEWETRON GmbH 2017
#pragma once

#include "otfft.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace OTFFT
{
namespace test
{

    class SineGenerator
    {
    public:
        explicit SineGenerator(const double sample_rate)
            : m_sample_rate (sample_rate)
            , m_frequency {0.0}
            , m_amplitude {1.0}
            , m_phase {0.0}
            , m_data ()
        {}

        SineGenerator& setFrequency(const double frequency)
        {
            m_frequency = frequency;
            return *this;
        }

        SineGenerator& setAmplitude(const double amplitude)
        {
            m_amplitude = amplitude;
            return *this;
        }

        SineGenerator& setPhase(const double phase)
        {
            m_phase = phase;
            return *this;
        }

        void generate(const std::size_t sample_count)
        {
            m_data.resize(sample_count);
            const double normalized_frequency = m_frequency / m_sample_rate;
            for (std::size_t sample_idx{0}; sample_idx < sample_count; ++sample_idx)
            {
                m_data[sample_idx] = m_amplitude * std::sin(2. * CONSTANT::PI * normalized_frequency * sample_idx + m_phase * 2. * CONSTANT::PI);
            }
        }

        double* data()
        {
            return m_data.data();
        }

    private:
        double m_sample_rate;
        double m_frequency;
        double m_amplitude;
        double m_phase;
        std::vector<double> m_data;
    };

}
}
