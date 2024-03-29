#pragma once

#include "tensor.h"

struct Batch
{
    Batch() { }
    Batch(NvU32 uBatch, TensorRef pInput, TensorRef pWantedOutput) :
        m_uBatch(uBatch), m_pInput(pInput), m_pWantedOutput(pWantedOutput)
    {
        m_pLoss = std::make_shared<Tensor>(m_pWantedOutput->getDims(), m_pWantedOutput->elemSize());
    }

    // returns the initial error (before progress was made)
    double makeMinimalProgress(struct NeuralNetwork& network, struct LossComputer& lossComputer,
        struct LearningRates& lr);

    TensorRef forwardPass(NeuralNetwork& network);
    
    NvU32 n() const
    {
        return m_pInput->n();
    }
    NvU32 getBatchIndex() const
    {
        return m_uBatch;
    }

    std::shared_ptr<Batch> cloneToPrecision(NvU32 elemSize)
    {
        auto p = std::make_shared<Batch>();
        p->m_uBatch = m_uBatch;
        p->m_pInput = m_pInput->cloneToPrecision(elemSize);
        p->m_pWantedOutput = m_pWantedOutput->cloneToPrecision(elemSize);
        p->m_pLoss = m_pLoss->cloneToPrecision(elemSize);
        return p;
    }

private:
    NvU32 m_uBatch = 0;

    TensorRef updateLoss(NeuralNetwork& network, LossComputer& lossComputer, double* pError = nullptr);

    TensorRef m_pInput, m_pWantedOutput, m_pLoss;
};
