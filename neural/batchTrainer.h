#pragma once

#include "tensor.h"
#include "learningRates.h"
#include "layerBatchData.h"

struct BatchTrainer
{
    void init(struct NeuralNetwork& network, NvU32 uBatch, TensorRef pInput, TensorRef pWantedOutput);

    void makeMinimalProgress(NeuralNetwork& network, struct LossComputer& lossComputer);

    void serialize(ISerializer& s)
    {
        m_lr.serialize(s);
    }

    void forwardPass(NeuralNetwork& network);
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

    LayerBatchData& get(NeuralNetwork& network, NvU32 uLayer);

private:
    NvU32 m_uBatch = 0;
    NvU32 n() const
    {
        nvAssert(m_pInput->n() > 0);
        return m_pInput->n();
    }
    TensorRef getInputs(NeuralNetwork &network, NvU32 uLayer)
    {
        return (uLayer == 0) ? m_pInput : get(network, uLayer - 1).m_pOutput;
    }
    void updateLoss(NeuralNetwork& network, LossComputer& lossComputer, float* pError = nullptr);

    TensorRef m_pInput, m_pWantedOutput;

    LearningRates m_lr;

    double m_fLRSum = 0;
    int m_nLRSamples = 0;
};
