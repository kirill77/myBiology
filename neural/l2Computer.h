#pragma once

#include "tensor.h"

struct LossComputer
{
    LossComputer()
    {
        // this limits maximam number of blocks we can have
        m_lossPerBlock.resize(1024 * 2);
    }

    void compute(Tensor<float> &outputs, Tensor<float> &wantedOutputs, Tensor<float> &outLoss, float *pAvgError = nullptr);

private:
    GPUBuffer<float> m_lossPerBlock;
};