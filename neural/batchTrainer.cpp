#include "batchTrainer.h"
#include "network.h"

void BatchTrainer::init(NeuralNetwork &network, NvU32 uBatch, TensorRef pInput, TensorRef pWantedOutput)
{
    (*this) = BatchTrainer();
    m_uBatch = uBatch;

    m_pInput = pInput;
    m_pWantedOutput = pWantedOutput;

    for (NvU32 uLayer = 0; uLayer < network.getNLayers(); ++uLayer)
    {
        network.getLayer(uLayer).allocateBatchData(uBatch, n());
    }

    m_lr.init(network.getNLayers());
}
void BatchTrainer::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer)
{
    forwardPass(network);
    float fError = 0;
    updateLoss(network, lossComputer, &fError);
    m_lr.setInitialError(fError);
    network.saveCurrentStateToBackup();

    for (NvU32 u = 0; u < m_lr.getNStepsToMake(); ++u)
    {
        backwardPass(network, lossComputer);
        forwardPass(network);
    }

    float fCurrentError = 0;
    updateLoss(network, lossComputer, &fCurrentError);
    bool bShouldRedo = true;
    m_lr.notifyNewError(fCurrentError, bShouldRedo);
    if (bShouldRedo)
    {
        network.restoreStateFromBackup();
        forwardPass(network);
    }
    else
    {
        network.saveCurrentStateToBackup();
    }
}
void BatchTrainer::updateLoss(NeuralNetwork &network, LossComputer& lossComputer, float *pErrorPtr)
{
    network.updateLoss(m_uBatch, *m_pWantedOutput, lossComputer, pErrorPtr);
}
void BatchTrainer::forwardPass(NeuralNetwork& network)
{
    for (NvU32 uLayer = 0; uLayer < network.getNLayers(); ++uLayer)
    {
        network.getLayer(uLayer).forward(getInputs(network, uLayer), get(network, uLayer), n());
    }
}
void BatchTrainer::backwardPass(NeuralNetwork& network, LossComputer& lossComputer)
{
    NvU32 nLayers = network.getNLayers();
    NvU32 uLayer = nLayers - 1;
    updateLoss(network, lossComputer);
    while (uLayer < nLayers)
    {
        TensorRef pInput = getInputs(network, uLayer);

        Tensor<float> *pPrevLoss = (uLayer == 0) ? nullptr : get(network, uLayer - 1).m_pLoss.get();
        float fBiasesLR = m_lr.getLearningRate(uLayer);
        float fWeightsLR = m_lr.getLearningRate(uLayer);
        m_fLRSum += fBiasesLR + fWeightsLR;
        m_nLRSamples += 2;
        network.getLayer(uLayer).backward(pInput,
            *get(network, uLayer).m_pLoss,
            fBiasesLR, fWeightsLR,
            get(network, uLayer), n(), pPrevLoss);
        --uLayer;
    }
}
LayerBatchData& BatchTrainer::get(NeuralNetwork& network, NvU32 uLayer)
{
    return network.getLayer(uLayer).m_batches[m_uBatch];
}