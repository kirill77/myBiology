#pragma once

#include "tensor.h"

struct Batch
{
    Batch(NvU32 uBatch, TensorRef pInput, TensorRef pWantedOutput) :
        m_uBatch(uBatch), m_pInput(pInput), m_pWantedOutput(pWantedOutput)
    {
    }

    // returns the initial error (before progress was made)
    float makeMinimalProgress(struct NeuralNetwork& network, struct LossComputer& lossComputer,
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

private:
    NvU32 m_uBatch = 0;

    TensorRef updateLoss(NeuralNetwork& network, LossComputer& lossComputer, double* pError = nullptr);

    TensorRef m_pInput, m_pWantedOutput;
};
