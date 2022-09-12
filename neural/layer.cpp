#include "layer.h"
#include "l2Computer.h"

void ILayer::allocateBatchData(NvU32 uBatch, NvU32 n)
{
    if (uBatch >= m_batches.size())
        m_batches.resize(uBatch + 1);
    auto& batchData = m_batches[uBatch];

    std::array<unsigned, 4> outputDims = m_outputDims;
    outputDims[0] = n;

    batchData.m_pLoss = std::make_shared<Tensor<float>>();
    batchData.m_pLoss->init(outputDims);

    batchData.m_pOutput = std::make_shared<Tensor<float>>();
    batchData.m_pOutput->init(outputDims);
}
void ILayer::updateLoss(NvU32 uBatch, Tensor<float>& wantedOutput, LossComputer& lossComputer, float* pErrorPtr)
{
    auto& bd = m_batches[uBatch];
    Tensor<float>& output = (*bd.m_pOutput);
    Tensor<float>& loss = (*bd.m_pLoss);
    lossComputer.compute(output, wantedOutput, loss, pErrorPtr);
}