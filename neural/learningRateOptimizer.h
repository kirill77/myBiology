#pragma once

#include <vector>
#include <basics/mybasics.h>
#include <basics/serializer.h>

struct LearningRateOptimizer
{
    NvU32 init(NvU32 nRates, struct NeuralNetwork& network);
    void makeMinimalProgress(NeuralNetwork& network);

    NvU32 notifyNewError(float fError, bool& bShouldRedo)
    {
        bool bLocalIncreaseOnPrevStep = m_bLocalIncreaseOnPrevStep;
        m_bLocalIncreaseOnPrevStep = false;
        m_nStepsMade += m_nStepsToMake;
        bool bErrorHasImproved = (isfinite(fError) && fError < m_fPrevError);
        bShouldRedo = !bErrorHasImproved;
        for ( ; ; ) // loop until the decision is made
        {
            if (bErrorHasImproved)
            {
                if (m_isGlobal)
                {
                    if (m_nStepsToMake < 128)
                    {
                        m_nStepsToMake *= 2;
                        break;
                    }
                }
                m_isGlobal = false;
                if (bLocalIncreaseOnPrevStep)
                {
                    RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                    float fNewErrorDecreaseRate = (m_fPrevError - fError) / m_nStepsToMake;
                    // did learning get slower?
                    if (fNewErrorDecreaseRate < m_fPrevErrorDecreaseRate)
                    {
                        nvAssert(r.m_fPrevValue < r.m_fValue);
                        r.m_fValue = r.m_fPrevValue;
                        r.m_uAttemptThreshold = std::min(1024u, std::max(2u, (NvU32)(r.m_uAttemptThreshold * 1.5)));
                        break;
                    }
                    // if the attempt was good - reward by decrementing the threshold
                    r.m_uAttemptThreshold = std::max(1u, r.m_uAttemptThreshold - 1);
                }
                // loop some limited amount trying to find which learning rate to increase
                for (NvU32 u = std::min((NvU32)m_pRatesInfo.size(), 8u); u != 0; --u)
                {
                    m_uLastAttemptedRate = (m_uLastAttemptedRate + 1) % m_pRatesInfo.size();
                    RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                    if (++r.m_uAttemptCounter >= r.m_uAttemptThreshold)
                    {
                        m_fPrevErrorDecreaseRate = (m_fPrevError - fError) / m_nStepsToMake;
                        r.m_uAttemptCounter = 0;
                        r.m_fPrevValue = r.m_fValue;
                        r.m_fValue *= 2;
                        m_bLocalIncreaseOnPrevStep = true;
                        break;
                    }
                }
                break;
            }
            else
            {
                if (bLocalIncreaseOnPrevStep)
                {
                    RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                    nvAssert(r.m_fPrevValue < r.m_fValue);
                    r.m_fValue = r.m_fPrevValue;
                    r.m_uAttemptThreshold = std::min(1024u, std::max(2u, r.m_uAttemptThreshold * 2));
                    break;
                }
                // if it wasn't global before - make rate increases less likely
                if (!m_isGlobal)
                {
                    for (NvU32 u = 0; u < m_pRatesInfo.size(); ++u)
                    {
                        RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                        r.m_uAttemptThreshold = std::min(1024u, std::max(2u, r.m_uAttemptThreshold * 2));
                    }
                }
                m_isGlobal = true;
                m_nStepsToMake = 1;
                for (NvU32 u = 0; u < m_pRatesInfo.size(); ++u)
                {
                    m_pRatesInfo[u].m_fValue /= 2;
                }
                break;
            }
        }
        if (bErrorHasImproved)
        {
            m_fPrevError = fError;
        }
        return m_nStepsToMake;
    }
    float getLearningRate(NvU32 uRate)
    {
        return m_pRatesInfo[uRate].m_fValue;
    }
    float getLastError() const
    {
        return m_fPrevError;
    }
    NvU32 getNStepsMade() const
    {
        return m_nStepsMade;
    }
    void serialize(ISerializer& s)
    {
        s.serializeStdArray("m_pRatesInfo", m_pRatesInfo);
        s.serializeSimpleType("m_isGlobal", m_isGlobal);
        s.serializeSimpleType("m_bLocalIncreaseOnPrevStep", m_bLocalIncreaseOnPrevStep);
        s.serializeSimpleType("m_fPrevError", m_fPrevError);
        s.serializeSimpleType("m_fPrevErrorDecreaseRate", m_fPrevErrorDecreaseRate);
        s.serializeSimpleType("m_nStepsToMake", m_nStepsToMake);
        s.serializeSimpleType("m_nStepsMade", m_nStepsMade);
        s.serializeSimpleType("m_uLastAttemptedRate", m_uLastAttemptedRate);
    }

private:
    struct RateInfo
    {
        float m_fPrevValue = 1, m_fValue = 1;
        NvU32 m_uAttemptThreshold = 1, m_uAttemptCounter = 0;
    };
    std::vector<RateInfo> m_pRatesInfo;

    bool m_isGlobal = true, m_bLocalIncreaseOnPrevStep = false;
    float m_fPrevError = std::numeric_limits<float>::max(), m_fPrevErrorDecreaseRate = 0;
    NvU32 m_nStepsToMake = 1, m_nStepsMade = 0, m_uLastAttemptedRate = 0;
};
