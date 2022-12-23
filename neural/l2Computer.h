#pragma once

#include "tensor.h"

struct LossComputer
{
    LossComputer()
    {
        // this limits maximam number of blocks we can have
        m_lossPerBlock.resize<float>(1024 * 2);
    }

    void compute(Tensor &outputs, Tensor &wantedOutputs, Tensor &outLoss, double *pAvgError = nullptr);

private:
    template <class T>
    void computeInternal(Tensor& outputs, Tensor& wantedOutputs, Tensor& outLoss, double* pAvgError);
    GPUBuffer m_lossPerBlock;
};