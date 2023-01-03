#include "batch.h"
#include "network.h"
#include "learningRates.h"

double Batch::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer,
    LearningRates &lr)
{
    network.allocateBatchData(m_uBatch, this->n());

    forwardPass(network);
    double fPreError = 0;
    updateLoss(network, lossComputer, &fPreError);
    lr.setInitialError(fPreError);
    network.saveCurrentStateToBackup();

    for (NvU32 u = 0; u < lr.getNStepsToMake(); ++u)
    {
        TensorRef pLoss = updateLoss(network, lossComputer);
        network.backwardPass(m_uBatch, pLoss.get(), lr);
        forwardPass(network);
    }

    double fCurrentError = 0;
    updateLoss(network, lossComputer, &fCurrentError);
    bool bShouldRedo = true;
    lr.notifyNewError(fCurrentError, bShouldRedo);
    if (bShouldRedo)
    {
        network.restoreStateFromBackup();
        forwardPass(network);
    }
    else
    {
        network.saveCurrentStateToBackup();
    }
    return fPreError;
}
TensorRef Batch::updateLoss(NeuralNetwork &network, LossComputer& lossComputer, double *pErrorPtr)
{
    network.updateLoss(m_uBatch, *m_pWantedOutput, lossComputer, m_pLoss, pErrorPtr);
    return m_pLoss;
}
TensorRef Batch::forwardPass(NeuralNetwork& network)
{
    return network.forwardPass(m_uBatch, m_pInput);
}
