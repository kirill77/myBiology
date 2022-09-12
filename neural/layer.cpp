#include "layer.h"

void ILayer::allocateBatchData(LayerBatchData& batchData, NvU32 n)
{
    std::array<unsigned, 4> outputDims = m_outputDims;
    outputDims[0] = n;

    batchData.m_pLoss = std::make_shared<Tensor<float>>();
    batchData.m_pLoss->init(outputDims);

    batchData.m_pOutput = std::make_shared<Tensor<float>>();
    batchData.m_pOutput->init(outputDims);
}