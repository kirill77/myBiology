#include "batchTrainer.h"
#include "network.h"
#include "learningRates.h"

void BatchTrainer::init(NeuralNetwork &network, NvU32 uBatch, TensorRef pInput, TensorRef pWantedOutput)
{
    (*this) = BatchTrainer();
    m_uBatch = uBatch;

    m_pInput = pInput;
    m_pWantedOutput = pWantedOutput;
    m_pLoss = std::make_shared<Tensor<float>>();
    m_pLoss->init(m_pWantedOutput->getDims());

    network.notifyBatchInited(uBatch, m_pInput->n());
}
float BatchTrainer::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer,
    LearningRates &lr)
{
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
    return fPreError;
}
TensorRef BatchTrainer::updateLoss(NeuralNetwork &network, LossComputer& lossComputer, float *pErrorPtr)
{
    network.updateLoss(m_uBatch, *m_pWantedOutput, lossComputer, *m_pLoss, pErrorPtr);
    return m_pLoss;
}
void BatchTrainer::forwardPass(NeuralNetwork& network)
{
    network.forwardPass(m_uBatch, m_pInput);
}
void BatchTrainer::backwardPass(NeuralNetwork& network, LossComputer& lossComputer, LearningRates &lr)
{
    TensorRef pLoss = updateLoss(network, lossComputer);
    network.backwardPass(m_uBatch, pLoss.get(), lr);
}
