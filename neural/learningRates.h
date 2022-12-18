#pragma once

#include "basics/serializer.h"

struct LearningRates
{
    void notifyNewError(float fError, bool& bShouldRedo);
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
    void init(NvU32 nRates)
    {
        m_isGlobal = true;
        m_pRatesInfo.resize(nRates);
    }
    void setInitialError(double fError)
    {
        nvAssert(isfinite(fError));
        m_fPrevError = fError;
    }
    float getLearningRate(NvU32 uRate) const
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
    NvU32 getNStepsToMake() const
    {
        return m_nStepsToMake;
    }

private:
    struct RateInfo
    {
        double m_fPrevValue = 1, m_fValue = 1;
        NvU32 m_uAttemptThreshold = 1, m_uAttemptCounter = 0;
    };
    std::vector<RateInfo> m_pRatesInfo;
    bool m_isGlobal = true, m_bLocalIncreaseOnPrevStep = false;
    double m_fPrevErrorDecreaseRate = 0;
    NvU32 m_uLastAttemptedRate = 0;

    double m_fPrevError = std::numeric_limits<float>::max();
    NvU32 m_nStepsToMake = 1, m_nStepsMade = 0;
};