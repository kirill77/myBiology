#include "batchTrainer.h"
#include "network.h"
#include "l2Computer.h"

void BatchTrainer::init(NeuralNetwork &network, std::vector<TensorRef> inputs, std::vector<TensorRef> wantedOutputs)
{
    (*this) = BatchTrainer();

    m_inputs = inputs;
    m_wantedOutputs = wantedOutputs;
    m_loss.init(m_wantedOutputs[0]->getDims());

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
    computeLoss(lossComputer, &fError);
    m_lr.setInitialError(fError);
    network.saveCurrentStateToBackup();

    for (NvU32 u = 0; u < m_lr.getNStepsToMake(); ++u)
    {
        backwardPass(network, lossComputer);
        forwardPass(network);
    }

    float fCurrentError = 0;
    computeLoss(lossComputer, &fCurrentError);
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
void BatchTrainer::computeLoss(LossComputer& lossComputer, float *pErrorPtr)
{
    const std::vector<TensorRef>& outputs = m_pLayerOutputs.rbegin()->m_outputs;
    nvAssert(outputs.size() == 1 && m_wantedOutputs.size() == 1);
    Tensor<float>& output = (*outputs[0]);
    Tensor<float>& wantedOutput = (*m_wantedOutputs[0]);
    lossComputer.compute(output, wantedOutput, m_loss, pErrorPtr);
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
    while (uLayer < nLayers)
    {
        std::vector<TensorRef>& inputs = getInputs(uLayer);

        // we don't need to compute deltaInputs for the layer 0
        std::vector<TensorRef>* pDeltaInputs = (uLayer == 0) ? nullptr : &m_pLayerOutputs[uLayer - 1].m_deltaOutputs;
        float fBiasesLR = m_lr.getLearningRate(uLayer);
        float fWeightsLR = m_lr.getLearningRate(uLayer);
        m_fLRSum += fBiasesLR + fWeightsLR;
        m_nLRSamples += 2;
        if (uLayer == nLayers - 1)
        {
            computeLoss(lossComputer);
            network.getLayer(uLayer).backward(inputs, m_loss,
                fBiasesLR, fWeightsLR, m_pLayerOutputs[uLayer], n(), pDeltaInputs);
        }
        else
        {
            network.getLayer(uLayer).backward(inputs,
                *m_pLayerOutputs[uLayer].m_deltaOutputs[0], fBiasesLR, fWeightsLR,
                m_pLayerOutputs[uLayer], n(), pDeltaInputs);
        }
        --uLayer;
    }
}

