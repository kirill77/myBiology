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
        batchData.m_pPrevLoss = std::make_shared<Tensor<float>>();
        batchData.m_pPrevLoss->init(inputDims);
    }

    batchData.m_pOutput = std::make_shared<Tensor<float>>();
    batchData.m_pOutput->init(outputDims);
}
void ILayer::updateLoss(NvU32 uBatch, Tensor<float>& wantedOutput, LossComputer& lossComputer, Tensor<float>& outLoss, float* pErrorPtr)
{
    auto& bd = m_batchesData.accessBatchData(uBatch);
    Tensor<float>& output = (*bd.m_pOutput);
    lossComputer.compute(output, wantedOutput, outLoss, pErrorPtr);
}
// functions used to check analytic derivative against numeric ones
NvU32 ILayer::getNParams() const
{
    return m_weights.size() + m_biases.size();
}
void ILayer::changeParam(NvU32 uParam, float fDeltaChange)
{
    NvU32 nWeights = m_weights.size();
    if (uParam < nWeights)
    {
        float fPrev = m_weights.getDeviceElem(uParam);
        m_weights.setDeviceElem(uParam, fPrev + fDeltaChange);
        return;
    }
    uParam -= nWeights;
    float fPrev = m_biases.getDeviceElem(uParam);
    m_biases.setDeviceElem(uParam, fPrev + fDeltaChange);
}
float ILayer::computeCurrentMinusBackup(NvU32 uParam)
{
    NvU32 nWeights = m_weights.size();
    float fCurrent = 0, fBackup = 0;
    if (uParam < nWeights)
    {
        fBackup = m_weightsBackup.getDeviceElem(uParam);
        fCurrent = m_weights.getDeviceElem(uParam);
    }
    else
    {
	    uParam -= nWeights;
        fBackup = m_biasesBackup.getDeviceElem(uParam);
        fCurrent = m_biases.getDeviceElem(uParam);
    }
    return fCurrent - fBackup;
}
