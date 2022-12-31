#pragma once

#include "tensor.h"

struct LayerBatchData
{
    void cloneRefsFrom(LayerBatchData& src, NvU32 elemSize)
    {
        m_beforeActivation = src.m_beforeActivation->cloneToPrecision(elemSize);
        m_pOutput = src.m_pOutput->cloneToPrecision(elemSize);
        m_pPrevLoss = src.m_pOutput->cloneToPrecision(elemSize);
        m_pPrevInput = src.m_pPrevInput->cloneToPrecision(elemSize);
    }
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

    std::shared_ptr<BatchDataContainer> cloneToPrecision(NvU32 elemSize)
    {
        auto p = std::make_shared<BatchDataContainer>();
        p->m_batchIndex = m_batchIndex;
        p->m_data.cloneRefsFrom(m_data, elemSize);
        return p;
    }

private:
    NvU32 m_batchIndex = 0xffffffff;
    LayerBatchData m_data;
};
