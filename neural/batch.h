#pragma once

#include "tensor.h"

struct Batch
{
    void init(struct NeuralNetwork& network, NvU32 uBatch, TensorRef pInput, TensorRef pWantedOutput);

    // returns the initial error (before progress was made)
    float makeMinimalProgress(NeuralNetwork& network, struct LossComputer& lossComputer,
        struct LearningRates& lr);

    void forwardPass(NeuralNetwork& network);
    void backwardPass(NeuralNetwork& network, LossComputer& lossComputer, LearningRates &lr);

private:
    NvU32 m_uBatch = 0;

    TensorRef updateLoss(NeuralNetwork& network, LossComputer& lossComputer, float* pError = nullptr);

    TensorRef m_pInput, m_pWantedOutput;
};
