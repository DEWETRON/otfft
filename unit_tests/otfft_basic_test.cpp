// Copyright (c) Dewetron 2017
#include "otfft.h"

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <thread>
#include <unordered_map>

namespace
{
    typedef struct MemberCapsule
    {
        MemberCapsule()
            : active(false)
            , fft()
        {
            BOOST_TEST_MESSAGE("MemberCapsule::MemberCapsule()");
        }
        MemberCapsule(MemberCapsule&& other)
            : active(std::move(other.active))
            , fft(std::move(other.fft))
        {
            BOOST_TEST_MESSAGE("MemberCapsule::MemberCapsule(MemberCapsule&& other)");
        }
        MemberCapsule& operator=(MemberCapsule&& other)
        {
            BOOST_TEST_MESSAGE("MemberCapsule::operator=(MemberCapsule&& other)");
            if (this != &other)
            {
                active = std::move(other.active);
                fft = std::move(other.fft);
            }
            return *this;
        }
        MemberCapsule(const MemberCapsule&)
        {
            BOOST_ERROR("MemberCapsule::MemberCapsule(const MemberCapsule&)");
        }
        MemberCapsule& operator=(const MemberCapsule&)
        {
            BOOST_ERROR("MemberCapsule::operator=(const MemberCapsule&)");
            return *this;
        }

        bool active;
        struct FFTMemberStruct
        {
            FFTMemberStruct()
                : rfft(nullptr)
                , cfft(nullptr)
                , dct(nullptr)
                , bluestein(nullptr)
            {
                BOOST_TEST_MESSAGE("FFTMemberStruct::FFTMemberStruct()");
            }
            FFTMemberStruct(FFTMemberStruct&& other)
                : rfft(std::move(other.rfft))
                , cfft(std::move(other.cfft))
                , dct(std::move(other.dct))
                , bluestein(std::move(other.bluestein))
            {
                BOOST_TEST_MESSAGE("FFTMemberStruct::FFTMemberStruct(FFTMemberStruct&& other)");
            }
            FFTMemberStruct& operator=(FFTMemberStruct&& other)
            {
                BOOST_TEST_MESSAGE("FFTMemberStruct::operator=(FFTMemberStruct&& other)");
                if (this != &other)
                {
                    rfft = std::move(other.rfft);
                    cfft = std::move(other.cfft);
                    dct = std::move(other.dct);
                    bluestein = std::move(other.bluestein);
                }
                return *this;
            }
            FFTMemberStruct(const FFTMemberStruct&)
            {
                BOOST_ERROR("FFTMemberStruct::FFTMemberStruct(const FFTMemberStruct&)");
            }
            FFTMemberStruct& operator=(const FFTMemberStruct&)
            {
                BOOST_ERROR("FFTMemberStruct::operator=(const FFTMemberStruct&)");
                return *this;
            }

            OTFFT::RealFFTPtr rfft;
            OTFFT::ComplexFFTPtr cfft;
            OTFFT::RealDCTPtr dct;
            OTFFT::ComplexFFTPtr bluestein;
        } fft;
    } MemberCapsule;

    class FFTClass;
    void ThreadHelper(FFTClass* fft_class);

    class FFTClass
    {
    public:
        explicit FFTClass(const std::size_t fft_size)
            : m_capsule()
            , m_thread()
            , m_fft_size(fft_size)
        {
            BOOST_TEST_MESSAGE("FFTClass::FFTClass()");
        }
        ~FFTClass()
        {
            BOOST_TEST_MESSAGE("FFTClass::~FFTClass()");
            m_capsule.active = false;
            if (m_thread.joinable())
            {
                m_thread.join();
            }
        }
        FFTClass(FFTClass&& other)
            : m_capsule(std::move(other.m_capsule))
            , m_thread(std::move(other.m_thread))
        {
            BOOST_TEST_MESSAGE("FFTClass::FFTClass(FFTClass&& other)");
        }
        FFTClass& operator=(FFTClass&& other)
        {
            BOOST_TEST_MESSAGE("FFTClass::operator=(FFTClass&& other)");
            if (this != &other)
            {
                m_capsule = std::move(other.m_capsule);
                m_thread = std::move(other.m_thread);
            }
            return *this;
        }

        void start()
        {
            BOOST_TEST_MESSAGE("FFTClass::start()");
            m_capsule.active = true;

            m_capsule.fft.cfft = OTFFT::Factory::createComplexFFT(static_cast<int>(m_fft_size));
            m_capsule.fft.bluestein = OTFFT::Factory::createBluesteinFFT(static_cast<int>(m_fft_size));

            if ((m_fft_size & 1) == 0)
            {
                m_capsule.fft.rfft = OTFFT::Factory::createRealFFT(static_cast<int>(m_fft_size));
                m_capsule.fft.dct = OTFFT::Factory::createDCT(static_cast<int>(m_fft_size));
            }

            m_thread = std::thread(ThreadHelper, this);
        }
        void stop()
        {
            BOOST_TEST_MESSAGE("FFTClass::stop()");
            m_capsule.active = false;
            if (m_thread.joinable())
            {
                m_thread.join();
            }
        }

        void runThread()
        {
            BOOST_TEST_MESSAGE("FFTClass::runThread - Enter");
            while (m_capsule.active)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

                m_capsule.fft.cfft->setup(static_cast<int>(m_fft_size));
                m_capsule.fft.bluestein->setup(static_cast<int>(m_fft_size));

                if ((m_fft_size & 1) == 0)
                {
                    m_capsule.fft.rfft->setup(static_cast<int>(m_fft_size));
                    m_capsule.fft.dct->setup(static_cast<int>(m_fft_size));
                }
            }
            BOOST_TEST_MESSAGE("FFTClass::runThread - Exit");
        }

    private:
        FFTClass(const FFTClass&)
        {
            BOOST_ERROR("FFTClass::FFTClass(const FFTClass&)");
        }
        FFTClass& operator=(const FFTClass&)
        {
            BOOST_ERROR("FFTClass::operator=(const FFTClass&)");
            return *this;
        }

    private:
        MemberCapsule m_capsule;
        std::thread m_thread;
        std::size_t m_fft_size;
    };

    typedef std::shared_ptr<FFTClass> FFTClassPtr;

    void ThreadHelper(FFTClass* fft_class)
    {
        fft_class->runThread();
    }
}

BOOST_AUTO_TEST_SUITE(otfft_basic_test)

BOOST_AUTO_TEST_CASE(TestFactory)
{
    BOOST_TEST_MESSAGE("Checking OTFFT factory");

    const std::size_t N = static_cast<std::size_t>(std::pow(2, 8));
    std::unordered_map<std::uint64_t, FFTClassPtr> fft_classes;

    for (std::size_t n = 8; n < N; n += static_cast<std::size_t>(std::max(1, static_cast<int>(std::log2(n)))))
    {
        {
            BOOST_TEST_MESSAGE("-> Create FFT Class with FFT order " << n);
            auto fft_class = std::make_shared<FFTClass>(n);
            fft_classes[n] = fft_class;
        }
        {
            BOOST_TEST_MESSAGE("-> Start Thread");
            auto fft_class = fft_classes[n];
            std::async(std::launch::async, [fft_class]{ fft_class->start(); }).wait();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        {
            BOOST_TEST_MESSAGE("-> Stop Thread");
            auto fft_class = fft_classes[n];
            std::async(std::launch::async, [fft_class]{ fft_class->stop(); }).wait();
        }
    }

    BOOST_TEST_MESSAGE("-> Release Resources");
    fft_classes.clear();

    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()
