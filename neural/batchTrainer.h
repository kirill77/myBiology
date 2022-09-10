#pragma once

#include "tensor.h"

struct LayerBatchData
{
    std::vector<TensorRef> m_deltaOutputs; // delta for the outputs we want to get from the previous layer
    std::vector<TensorRef> m_beforeActivation; // this is the m_outputs before activation function
    std::vector<TensorRef> m_outputs; // output of this layer
};

struct LearningRates
{
    NvU32 notifyNewError(float fError, bool& bShouldRedo);
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
    void setInitialError(float fError)
    {
        nvAssert(fError == m_fPrevError || m_nStepsMade == 0);
        m_fPrevError = fError;
        nvAssert(isfinite(m_fPrevError));
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
        float m_fPrevValue = 1, m_fValue = 1;
        NvU32 m_uAttemptThreshold = 1, m_uAttemptCounter = 0;
    };
    std::vector<RateInfo> m_pRatesInfo;
    bool m_isGlobal = true, m_bLocalIncreaseOnPrevStep = false;
    float m_fPrevErrorDecreaseRate = 0;
    NvU32 m_uLastAttemptedRate = 0;

    float m_fPrevError = std::numeric_limits<float>::max();
    NvU32 m_nStepsToMake = 1, m_nStepsMade = 0;
};

struct BatchTrainer
{
    void init(struct NeuralNetwork& network, std::vector<TensorRef> inputs, std::vector<TensorRef> wantedOutputs);
    void makeMinimalProgress(NeuralNetwork& network, struct LossComputer& lossComputer);

    void computeLoss(LossComputer& lossComputer, float *pError = nullptr);

    void serialize(ISerializer& s)
    {
        m_lr.serialize(s);
    }
    NvU32 n() const
    {
        nvAssert(m_inputs[0]->n() > 0);
        return m_inputs[0]->n();
    }
    LayerBatchData& accessLayerData(NvU32 uLayer)
    {
        return m_pLayerOutputs[uLayer];
    }
    std::vector<TensorRef>& getInputs(NvU32 uLayer)
    {
        return (uLayer == 0) ? m_inputs : accessLayerData(uLayer - 1).m_outputs;
    }
    void backwardPass(NeuralNetwork& network, LossComputer& lossComputer);
    double computeAvgLRStats() const
    {
        return (m_nLRSamples == 0) ? 0 : m_fLRSum / m_nLRSamples;
    }
    void resetAvgLRStats()
    {
        m_fLRSum = 0;
        m_nLRSamples = 0;
    }
    const LearningRates& getLR() const
    {
        return m_lr;
    }

    std::vector<TensorRef> m_wantedOutputs;
    Tensor<float> m_loss;

private:
    std::vector<TensorRef> m_inputs;
    std::vector<LayerBatchData> m_pLayerOutputs;

    LearningRates m_lr;

    double m_fLRSum = 0;
    int m_nLRSamples = 0;
};
