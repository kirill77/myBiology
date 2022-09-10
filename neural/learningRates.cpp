#include "learningRates.h"

NvU32 LearningRates::notifyNewError(float fError, bool& bShouldRedo)
{
    bool bLocalIncreaseOnPrevStep = m_bLocalIncreaseOnPrevStep;
    m_bLocalIncreaseOnPrevStep = false;
    m_nStepsMade += m_nStepsToMake;
    bool bErrorHasImproved = (isfinite(fError) && fError < m_fPrevError);
    bShouldRedo = !bErrorHasImproved;
    for (; ; ) // loop until the decision is made
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
