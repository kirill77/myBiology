#pragma once

#include <vector>
#include <basics/mybasics.h>

struct LearningRateOptimizer
{
    NvU32 init(NvU32 nLayers, float fInitialError)
    {
        nvAssert(isfinite(fInitialError));
        (*this) = LearningRateOptimizer();
        m_layerRates.resize(nLayers);
        std::fill(m_layerRates.begin(), m_layerRates.end(), 1.f);
        m_layerRatesMax.resize(nLayers);
        std::fill(m_layerRatesMax.begin(), m_layerRatesMax.end(), 1.f);

        // that's because we go over layers in round-robin fashion and we want to remember information about each
        // layer on previous iteration
        m_stateHistory.resize(nLayers + 1);

        for (NvU32 u = 0; u < m_stateHistory.size(); ++u)
        {
            m_stateHistory[u].init(nLayers);
        }

        pushNewState(LRO_STATE_INIT);
        State& newState = getState();
        m_fLargestAcceptableError = fInitialError;
        return newState.m_nStepsToMake;
    }
    NvU32 notifyNewError(float fError, bool& bShouldRedo)
    {
        State& prevState = getState();
        prevState.m_fError = fError;
        bool isErrorAccepted = isfinite(fError) && fError < m_fLargestAcceptableError;

#if 0
        // an additional checks for LRO_STATE_LOCAL_ASCENT
        if (isErrorAccepted && prevState.m_state == LRO_STATE_LOCAL_ASCENT)
        {
            const State& sameLayerOldState = getState((NvU32)m_layerRates.size());
            if (sameLayerOldState.m_state == LRO_STATE_LOCAL_ASCENT)
            {
                nvAssert(sameLayerOldState.m_layerIndex == prevState.m_layerIndex);
                if (prevState.m_fError > sameLayerOldState.m_fError)
                {
                    float f = (prevState.m_fError - sameLayerOldState.m_fError) / (m_fLargestAcceptableError - sameLayerOldState.m_fError);
                    if (f > 0.25f)
                    {
                        isErrorAccepted = false;
                    }
                }
            }
        }
#endif

        if (isErrorAccepted)
        {
            m_nBadSteps = 0;
            switch (prevState.m_state)
            {
            case LRO_STATE_GLOBAL_DESCENT:
                // we were decreasing learning rate globally and got acceptable error
                // now start increasing learning rate for individual layers
                pushNewState(LRO_STATE_LOCAL_ASCENT);
                break;
            case LRO_STATE_LOCAL_ASCENT:
                // as long as error is acceptable - keep increasing layers we can increase
                pushNewState(LRO_STATE_LOCAL_ASCENT);
                break;
            case LRO_STATE_DONE:
                pushNewState(LRO_STATE_DONE);
                break;
            default:
                nvAssert(false);
                break;
            }
        }
        else
        {
            ++m_nBadSteps;
            switch (prevState.m_state)
            {
            case LRO_STATE_DONE:
                // make allowed rates a bit lower
                for (NvU32 u = 0; u < m_layerRatesMax.size(); ++u)
                {
                    m_layerRatesMax[u] *= 0.9f;
                }
                // pass-through intentional
            case LRO_STATE_INIT:
            case LRO_STATE_GLOBAL_DESCENT:
                // continue global descent until we arrive to acceptable error
                pushNewState(LRO_STATE_GLOBAL_DESCENT);
                break;
            case LRO_STATE_LOCAL_ASCENT:
                // remember not to go higher than that next time
                m_layerRatesMax[prevState.m_layerIndex] = std::min(m_layerRatesMax[prevState.m_layerIndex], m_layerRates[prevState.m_layerIndex]);
                backUpPushedState(); // unsuccessfull local accent - back it up
                if (m_nBadSteps > m_layerRates.size())
                {
                    // for a while we couldn't find a layer on which we could increase learning rate - time to stop
                    pushNewState(LRO_STATE_DONE);
                }
                else
                {
                    // previous LOCAL_ASCENT attempt was unsuccessful - try another layer
                    pushNewState(LRO_STATE_LOCAL_ASCENT);
                }
                break;
            default:
                nvAssert(false);
                break;
            }
        }

        State& newState = getState();
        bShouldRedo = !isErrorAccepted;
        if (isErrorAccepted)
        {
            m_fLargestAcceptableError = std::min(m_fLargestAcceptableError, fError);
            newState.m_nStepsToMake = (newState.m_state == LRO_STATE_DONE) ? std::min(prevState.m_nStepsToMake * 2, 128u) : 1;
        }
        else
        {
            newState.m_nStepsToMake = 1;
        }
        return newState.m_nStepsToMake;
    }
    float getLearningRate(NvU32 uLayer)
    {
        float fLearningRate = m_layerRates[uLayer];
        return fLearningRate;
    }
    void serialize(ISerializer& s)
    {
        s.serializeStdArray("m_layerRates", m_layerRates);
        s.serializeStdArray("m_layerRatesMax", m_layerRatesMax);
        s.serializeSimpleType("m_curHistIndex", m_curHistIndex);
        s.serializeSimpleType("m_nBadSteps", m_nBadSteps);
        s.serializeSimpleType("m_fLargestAcceptableError", m_fLargestAcceptableError);
        
        {
            std::shared_ptr<Indent> pIndent = s.pushIndent("m_stateHistory");
            s.serializeArraySize("m_stateHistory", m_stateHistory);
            for (NvU32 u = 0; u < m_stateHistory.size(); ++u)
            {
                m_stateHistory[u].serialize(s);
            }
        }
    }

private:
    std::vector<float> m_layerRates, m_layerRatesMax;
    NvU32 m_curHistIndex = 0xffffffff, m_nBadSteps = 0;
    float m_fLargestAcceptableError = std::numeric_limits<float>::max();
    enum LRO_STATE { LRO_STATE_INIT, LRO_STATE_GLOBAL_DESCENT, LRO_STATE_LOCAL_ASCENT, LRO_STATE_DONE };
    struct State
    {
        State() { }
        void init(NvU32 nLayers)
        {
            (*this) = State();
            m_layerRatesBackup.resize(nLayers);
            std::fill(m_layerRatesBackup.begin(), m_layerRatesBackup.end(), 1.f);
        }
        void serialize(ISerializer& s)
        {
            std::shared_ptr<Indent> pIndent = s.pushIndent("State");
            s.serializeSimpleType("m_state", m_state);
            s.serializeStdArray("m_layerRatesBackup", m_layerRatesBackup);
            s.serializeSimpleType("m_layerIndex", m_layerIndex);
            s.serializeSimpleType("m_nStepsToMake", m_nStepsToMake);
            s.serializeSimpleType("m_fError", m_fError);
        }

        LRO_STATE m_state = LRO_STATE_INIT;
        std::vector<float> m_layerRatesBackup;
        NvU32 m_layerIndex = 0xffffffff, m_nStepsToMake = 1, m_index = 0xffffffff;
        float m_fError = -1; // error we got
    };
    std::vector<State> m_stateHistory;
    void pushNewState(LRO_STATE state)
    {
        float fMultiplier = -1;
        NvU32 layerIndex = 0xffffffff;
        switch (state)
        {
        case LRO_STATE_INIT:
        case LRO_STATE_DONE:
            fMultiplier = 1;
            layerIndex = 0; // just an optimization - index doesn't matter here
            break;
        case LRO_STATE_GLOBAL_DESCENT:
            fMultiplier = 0.5;
            break;
        case LRO_STATE_LOCAL_ASCENT:
            layerIndex = getState().m_layerIndex; // derive from previous layer index
            // round-robin between layers. find the first layer which has learning rate < 1 and increase it
            for (NvU32 u = 0; ; )
            {
                layerIndex = (layerIndex + 1) % m_layerRates.size();
                fMultiplier = std::min(2.f, m_layerRatesMax[layerIndex] / m_layerRates[layerIndex]);
                if (fMultiplier > 1.1f) // can we increase learning rate substantially?
                {
                    break;
                }
                if (++u >= m_layerRates.size())
                {
                    fMultiplier = 1; // didn't find the suitable layer? leave the index unchanged
                    state = LRO_STATE_DONE;
                    break;
                }
            }
            break;
        default:
            nvAssert(false);
            break;
        }
        ++m_curHistIndex;
        State& topState = getState();
        topState.m_state = state;
        topState.m_layerIndex = layerIndex;
        topState.m_index = m_curHistIndex;
        if (layerIndex == 0xffffffff)
        {
            for (NvU32 u = 0; u < m_layerRates.size(); ++u)
            {
                topState.m_layerRatesBackup[u] = m_layerRates[u];
                m_layerRates[u] *= fMultiplier;
            }
        }
        else
        {
            topState.m_layerRatesBackup[layerIndex] = m_layerRates[layerIndex];
            m_layerRates[layerIndex] *= fMultiplier;
        }
    }
    const State& getState(NvU32 goBackIndex = 0) const
    {
        nvAssert(goBackIndex < m_stateHistory.size());
        return m_stateHistory[(m_curHistIndex + m_stateHistory.size() - goBackIndex) % m_stateHistory.size()];
    }
    State& getState(NvU32 goBackIndex = 0)
    {
        nvAssert(goBackIndex < m_stateHistory.size());
        return m_stateHistory[(m_curHistIndex + m_stateHistory.size() - goBackIndex) % m_stateHistory.size()];
    }
    void backUpPushedState()
    {
        State& topState = getState();
        nvAssert(topState.m_layerIndex != 0xffffffff);
        m_layerRates[topState.m_layerIndex] = topState.m_layerRatesBackup[topState.m_layerIndex];
    }
};
