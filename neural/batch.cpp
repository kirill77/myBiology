#include "batch.h"
#include "network.h"
#include "learningRates.h"

void Batch::init(NeuralNetwork &network, NvU32 uBatch, TensorRef pInput, TensorRef pWantedOutput)
{
    (*this) = Batch();
    m_uBatch = uBatch;

    m_pInput = pInput;
    m_pWantedOutput = pWantedOutput;
}
float Batch::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer,
    LearningRates &lr)
{
    network.allocateBatchData(m_uBatch, m_pInput->n());
    forwardPass(network);
    float fPreError = 0;
    updateLoss(network, lossComputer, &fPreError);
    lr.setInitialError(fPreError);
    network.saveCurrentStateToBackup();

    for (NvU32 u = 0; u < lr.getNStepsToMake(); ++u)
    {
        backwardPass(network, lossComputer, lr);
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
    network.freeBatchData(m_uBatch);
    return fPreError;
}
TensorRef Batch::updateLoss(NeuralNetwork &network, LossComputer& lossComputer, float *pErrorPtr)
{
    return network.updateLoss(m_uBatch, *m_pWantedOutput, lossComputer, pErrorPtr);
}
void Batch::forwardPass(NeuralNetwork& network)
{
    network.forwardPass(m_uBatch, m_pInput);
}
void Batch::backwardPass(NeuralNetwork& network, LossComputer& lossComputer, LearningRates &lr)
{
    TensorRef pLoss = updateLoss(network, lossComputer);
    network.backwardPass(m_uBatch, pLoss.get(), lr);
}
