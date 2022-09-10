#pragma once

#include "tensor.h"

struct LayerBatchData
{
    std::vector<TensorRef> m_deltaOutputs; // delta for the outputs we want to get from the previous layer
    std::vector<TensorRef> m_beforeActivation; // this is the m_outputs before activation function
    std::vector<TensorRef> m_outputs; // output of this layer
};
