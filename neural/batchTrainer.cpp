#include "batchTrainer.h"
#include "network.h"

void BatchTrainer::init(NeuralNetwork &network, std::vector<TensorRef> inputs, std::vector<TensorRef> wantedOutputs)
{
    (*this) = BatchTrainer();

    m_inputs = inputs;
    m_wantedOutputs = wantedOutputs;

    m_pLayerOutputs.resize(network.getNLayers());
    for (NvU32 u = 0; u < m_pLayerOutputs.size(); ++u)
    {
        network.getLayer(u).allocateBatchData(*this);
    }

    m_isGlobal = true;
    m_pRatesInfo.resize(network.getNLayers());
}
void BatchTrainer::makeMinimalProgress(NeuralNetwork& network, L2Computer &l2Computer)
{
    network.forwardPass(*this);
    m_fPrevError = computeCurrentError(l2Computer);
    nvAssert(isfinite(m_fPrevError));
    network.saveCurrentStateToBackup();

    network.makeSteps(m_nStepsToMake, *this);
    float fCurrentError = computeCurrentError(l2Computer);
    bool bShouldRedo = true;
    notifyNewError(fCurrentError, bShouldRedo);
    if (bShouldRedo)
    {
        network.restoreStateFromBackup();
        network.forwardPass(*this);
    }
    else
    {
        m_fPrevError = fCurrentError;
        network.saveCurrentStateToBackup();
    }
}

float BatchTrainer::computeCurrentError(struct L2Computer& l2Computer)
{
    const std::vector<TensorRef>& outputs = m_pLayerOutputs.rbegin()->m_outputs;
    nvAssert(outputs.size() == m_wantedOutputs.size());
    for (NvU32 uTensor = 0; uTensor < outputs.size(); ++uTensor)
    {
        Tensor<float>& output = (*outputs[uTensor]);
        Tensor<float>& wantedOutput = (*m_wantedOutputs[uTensor]);
        l2Computer.accumulateL2Error(output, wantedOutput, (uTensor == 0) ? L2_MODE_RESET : L2_MODE_ADD);
    }
    float fError = l2Computer.getAccumulatedError();
    return fError;
}
