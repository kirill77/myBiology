#include "batchTrainer.h"
#include "network.h"

void BatchTrainer::init(NvU32 nLayers, NvU32 nRates, NeuralNetwork& network)
{
    (*this) = BatchTrainer();

    m_isGlobal = true;
    m_pRatesInfo.resize(nRates);

    network.forwardPass();
    m_fPrevError = network.computeCurrentError();
    nvAssert(isfinite(m_fPrevError));
    network.saveCurrentStateToBackup();

    m_pLayerOutputs.resize(nLayers);
}
void BatchTrainer::makeMinimalProgress(NeuralNetwork& network)
{
    network.makeSteps(m_nStepsToMake, *this);
    float fCurrentError = network.computeCurrentError();
    bool bShouldRedo = true;
    notifyNewError(fCurrentError, bShouldRedo);
    if (bShouldRedo)
    {
        network.restoreStateFromBackup();
        network.forwardPass();
    }
    else
    {
        m_fPrevError = fCurrentError;
        network.saveCurrentStateToBackup();
    }
}
