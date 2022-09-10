#pragma once

#include "tensor.h"
#include "learningRates.h"

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
