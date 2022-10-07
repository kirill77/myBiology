#pragma once

#include "tensor.h"

struct LayerBatchData
{
    TensorRef m_beforeActivation; // this is the m_outputs before activation function
    TensorRef m_pOutput, m_pPrevLoss, m_pPrevInput;
};

struct BatchDataContainer
{
    LayerBatchData& allocateBatchData(NvU32 uBatch)
    {
        m_batchIndex = uBatch;
        m_data = LayerBatchData();
        return m_data;
    }
    LayerBatchData& accessBatchData(NvU32 uBatch)
    {
        nvAssert(uBatch == m_batchIndex);
        return m_data;
    }
private:
    NvU32 m_batchIndex = 0xffffffff;
    LayerBatchData m_data;
};
