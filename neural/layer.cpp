#include "layer.h"
#include "l2Computer.h"

void ILayer::allocateBatchData(NvU32 uBatch, NvU32 n, bool isFirstLayer)
{
    LayerBatchData& batchData = m_batchesData.allocateBatchData(uBatch);

    std::array<unsigned, 4> outputDims = m_outputDims;
    outputDims[0] = n;
    std::array<unsigned, 4> inputDims = m_inputDims;
    inputDims[0] = n;

    if (!isFirstLayer)
    {
        batchData.m_pPrevLoss = std::make_shared<Tensor>(inputDims, sizeof(float));
    }

    batchData.m_pOutput = std::make_shared<Tensor>(outputDims, sizeof(float));
}
void ILayer::updateLoss(NvU32 uBatch, Tensor& wantedOutput, LossComputer& lossComputer, Tensor& outLoss, double* pErrorPtr)
{
    auto& bd = m_batchesData.accessBatchData(uBatch);
    Tensor& output = (*bd.m_pOutput);
    lossComputer.compute(output, wantedOutput, outLoss, pErrorPtr);
}
// functions used to check analytic derivative against numeric ones
NvU32 ILayer::getNTrainableParams() const
{
    return m_weights.size() + m_biases.size();
}
double ILayer::getTrainableParam(NvU32 uParam)
{
    NvU32 nWeights = m_weights.size();
    if (uParam < nWeights)
    {
        double f = m_weights.autoReadElem<float>(uParam);
        return f;
    }
    uParam -= nWeights;
    double f = m_biases.autoReadElem<float>(uParam);
    return f;
}
void ILayer::setTrainableParam(NvU32 uParam, double fValue)
{
    NvU32 nWeights = m_weights.size();
    if (uParam < nWeights)
    {
        m_weights.autoWriteElem<float>(uParam, fValue);
        return;
    }
    uParam -= nWeights;
    m_biases.autoWriteElem<float>(uParam, fValue);
}
