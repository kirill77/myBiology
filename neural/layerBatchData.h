#pragma once

#include "tensor.h"

struct LayerBatchData
{
    TensorRef m_beforeActivation; // this is the m_outputs before activation function
    TensorRef m_pOutput, m_pPrevLoss, m_pPrevInput;
};
