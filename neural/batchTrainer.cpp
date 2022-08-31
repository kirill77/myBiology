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

    network.forwardPass(*this);
    m_fPrevError = network.computeCurrentError(*this);
    nvAssert(isfinite(m_fPrevError));
    network.saveCurrentStateToBackup();
}
void BatchTrainer::makeMinimalProgress(NeuralNetwork& network)
{
    network.makeSteps(m_nStepsToMake, *this);
    float fCurrentError = network.computeCurrentError(*this);
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
