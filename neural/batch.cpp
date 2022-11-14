#include "batch.h"
#include "network.h"
#include "learningRates.h"

float Batch::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer,
    LearningRates &lr)
{
    forwardPass(network);
    float fPreError = 0;
    updateLoss(network, lossComputer, &fPreError);
    lr.setInitialError(fPreError);
    network.saveCurrentStateToBackup();

    for (NvU32 u = 0; u < lr.getNStepsToMake(); ++u)
    {
        TensorRef pLoss = updateLoss(network, lossComputer);
        backwardPass(network, *pLoss, lr);
        forwardPass(network);
    }

    float fCurrentError = 0;
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
TensorRef Batch::updateLoss(NeuralNetwork &network, LossComputer& lossComputer, float *pErrorPtr)
{
    return network.updateLoss(m_uBatch, *m_pWantedOutput, lossComputer, pErrorPtr);
}
TensorRef Batch::forwardPass(NeuralNetwork& network)
{
    return network.forwardPass(m_uBatch, m_pInput);
}
void Batch::backwardPass(NeuralNetwork& network, Tensor &loss, LearningRates &lr)
{
    network.backwardPass(m_uBatch, &loss, lr);
}
