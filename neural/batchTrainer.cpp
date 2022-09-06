#include "batchTrainer.h"
#include "network.h"

void BatchTrainer::init(NeuralNetwork &network, std::vector<TensorRef> inputs, std::vector<TensorRef> wantedOutputs)
{
    (*this) = BatchTrainer();

    m_inputs = inputs;
    m_wantedOutputs = wantedOutputs;
    m_loss.init(m_wantedOutputs[0]->getDims());

    m_pLayerOutputs.resize(network.getNLayers());
    for (NvU32 u = 0; u < m_pLayerOutputs.size(); ++u)
    {
        network.getLayer(u).allocateBatchData(*this);
    }

    m_isGlobal = true;
    m_pRatesInfo.resize(network.getNLayers());
}
void BatchTrainer::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer)
{
    network.forwardPass(*this);
    m_fPrevError = computeCurrentError(lossComputer);
    nvAssert(isfinite(m_fPrevError));
    network.saveCurrentStateToBackup();

    network.makeSteps(m_nStepsToMake, *this);
    float fCurrentError = computeCurrentError(lossComputer);
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

float BatchTrainer::computeCurrentError(LossComputer& lossComputer)
{
    const std::vector<TensorRef>& outputs = m_pLayerOutputs.rbegin()->m_outputs;
    nvAssert(outputs.size() == 1 && m_wantedOutputs.size() == 1);
    Tensor<float>& output = (*outputs[0]);
    Tensor<float>& wantedOutput = (*m_wantedOutputs[0]);
    float fError = 0;
    lossComputer.compute(output, wantedOutput, m_loss, &fError);
    return fError;
}
