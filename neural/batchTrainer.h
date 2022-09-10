#pragma once

#include "tensor.h"

struct LayerBatchData
{
    std::vector<TensorRef> m_deltaOutputs; // delta for the outputs we want to get from the previous layer
    std::vector<TensorRef> m_beforeActivation; // this is the m_outputs before activation function
    std::vector<TensorRef> m_outputs; // output of this layer
};

struct BatchTrainer
{
    void init(struct NeuralNetwork& network, std::vector<TensorRef> inputs, std::vector<TensorRef> wantedOutputs);
    void makeMinimalProgress(NeuralNetwork& network, struct LossComputer& lossComputer);

    void computeLoss(LossComputer& lossComputer, float *pError = nullptr);

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
    std::vector<TensorRef> m_wantedOutputs;
    Tensor<float> m_loss;

private:
    std::vector<TensorRef> m_inputs;
    std::vector<LayerBatchData> m_pLayerOutputs;
    NvU32 notifyNewError(float fError, bool& bShouldRedo);
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
