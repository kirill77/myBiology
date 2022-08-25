#include "batchTrainer.h"
#include "network.h"

void BatchTrainer::init(std::vector<std::shared_ptr<ILayer>> &pLayers, NvU32 nRates, NeuralNetwork& network)
{
    (*this) = BatchTrainer();

    m_pLayerOutputs.resize(pLayers.size());
    for (NvU32 u = 0; u < m_pLayerOutputs.size(); ++u)
    {
        pLayers[u]->allocateBatchData(*this);
    }

    m_isGlobal = true;
    m_pRatesInfo.resize(nRates);

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
