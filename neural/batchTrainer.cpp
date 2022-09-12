#include "batchTrainer.h"
#include "network.h"
#include "l2Computer.h"

void BatchTrainer::init(NeuralNetwork &network, TensorRef pInput, TensorRef pWantedOutput)
{
    (*this) = BatchTrainer();

    m_pInput = pInput;
    m_pWantedOutput = pWantedOutput;

    m_pLayerOutputs.resize(network.getNLayers());
    for (NvU32 u = 0; u < m_pLayerOutputs.size(); ++u)
    {
        network.getLayer(u).allocateBatchData(m_pLayerOutputs[u], n());
    }

    m_lr.init(network.getNLayers());
}
void BatchTrainer::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer)
{
    forwardPass(network);
    float fError = 0;
    updateLoss(lossComputer, &fError);
    m_lr.setInitialError(fError);
    network.saveCurrentStateToBackup();

    for (NvU32 u = 0; u < m_lr.getNStepsToMake(); ++u)
    {
        backwardPass(network, lossComputer);
        forwardPass(network);
    }

    float fCurrentError = 0;
    updateLoss(lossComputer, &fCurrentError);
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
void BatchTrainer::updateLoss(LossComputer& lossComputer, float *pErrorPtr)
{
    Tensor<float>& output = (*m_pLayerOutputs.rbegin()->m_pOutput);
    Tensor<float>& wantedOutput = (*m_pWantedOutput);
    Tensor<float>& loss = (*m_pLayerOutputs.rbegin()->m_pLoss);
    lossComputer.compute(output, wantedOutput, loss, pErrorPtr);
}
void BatchTrainer::forwardPass(NeuralNetwork& network)
{
    for (NvU32 uLayer = 0; uLayer < network.getNLayers(); ++uLayer)
    {
        network.getLayer(uLayer).forward(getInputs(uLayer), m_pLayerOutputs[uLayer], n());
    }
}
void BatchTrainer::backwardPass(NeuralNetwork& network, LossComputer& lossComputer)
{
    NvU32 nLayers = network.getNLayers();
    NvU32 uLayer = nLayers - 1;
    updateLoss(lossComputer);
    while (uLayer < nLayers)
    {
        TensorRef pInput = getInputs(uLayer);

        Tensor<float> *pPrevLoss = (uLayer == 0) ? nullptr : m_pLayerOutputs[uLayer - 1].m_pLoss.get();
        float fBiasesLR = m_lr.getLearningRate(uLayer);
        float fWeightsLR = m_lr.getLearningRate(uLayer);
        m_fLRSum += fBiasesLR + fWeightsLR;
        m_nLRSamples += 2;
        network.getLayer(uLayer).backward(pInput,
            *m_pLayerOutputs[uLayer].m_pLoss,
            fBiasesLR, fWeightsLR,
            m_pLayerOutputs[uLayer], n(), pPrevLoss);
        --uLayer;
    }
}

