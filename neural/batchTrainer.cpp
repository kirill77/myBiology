#include "learningRateOptimizer.h"
#include "network.h"

NvU32 LearningRateOptimizer::init(NvU32 nRates, NeuralNetwork& network)
{
    (*this) = LearningRateOptimizer();

    m_isGlobal = true;
    m_pRatesInfo.resize(nRates);

    network.forwardPass();
    m_fPrevError = network.computeCurrentError();
    nvAssert(isfinite(m_fPrevError));
    network.saveCurrentStateToBackup();

    return m_nStepsToMake;
}
void LearningRateOptimizer::makeMinimalProgress(NeuralNetwork& network)
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
