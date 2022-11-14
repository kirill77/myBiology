#pragma once

#include "tensor.h"

struct LossComputer
{
    LossComputer()
    {
        // this limits maximam number of blocks we can have
        m_lossPerBlock.resize<float>(1024 * 2);
    }

    void compute(Tensor &outputs, Tensor &wantedOutputs, Tensor &outLoss, float *pAvgError = nullptr);

private:
    GPUBuffer m_lossPerBlock;
};