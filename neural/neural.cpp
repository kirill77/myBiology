#include "network.h"
#include "neuralTest.h"
#include "learningRates.h"
#include "l2Computer.h"

NeuralNetwork::NeuralNetwork()
{
    nvAssert(NeuralTest::isTested());
}

std::shared_ptr<ILayer> ILayer::createLayer(LAYER_TYPE layerType)
{
    std::shared_ptr<ILayer> pLayer;
    switch (layerType)
    {
    case LAYER_TYPE_FCL_IDENTITY:
        pLayer = std::make_shared<FullyConnectedLayer<ACTIVATION_IDENTITY,
            ACTIVATION_IDENTITY>>();
        break;
    case LAYER_TYPE_FCL_MIRRORED:
        pLayer = std::make_shared<FullyConnectedLayer<ACTIVATION_RELU,
            ACTIVATION_MRELU>>();
        break;
    default:
        nvAssert(false);
        return nullptr;
    }
    nvAssert(pLayer->m_type == layerType);
    return pLayer;
}

void NeuralNetwork::backwardPass(NvU32 uBatch, Tensor* pLoss, struct LearningRates& lr)
{
    NvU32 uLayer = (NvU32)m_pLayers.size() - 1;
    while (uLayer < m_pLayers.size())
    {
        double fBiasesLR = lr.getLearningRate(uLayer);
        double fWeightsLR = lr.getLearningRate(uLayer);
        m_fLRSum += fBiasesLR + fWeightsLR;
        m_nLRSamples += 2;
        pLoss = m_pLayers[uLayer]->backward(uBatch, *pLoss,
            fBiasesLR, fWeightsLR);
        --uLayer;
    }
}

NvU32 NeuralNetwork::getNTrainableParams() const
{
    NvU32 nParams = 0;
    for (NvU32 u = 0; u < m_pLayers.size(); ++u)
    {
        nParams += m_pLayers[u]->getNTrainableParams();
    }
    return nParams;
}
double NeuralNetwork::getTrainableParam(NvU32 uParam)
{
    for (NvU32 u = 0; u < m_pLayers.size(); ++u)
    {
        if (uParam < m_pLayers[u]->getNTrainableParams())
            return m_pLayers[u]->getTrainableParam(uParam);
        uParam -= m_pLayers[u]->getNTrainableParams();
    }
    nvAssert(false); // no such param
    return 0;
}
void NeuralNetwork::setTrainableParam(NvU32 uParam, double fValue)
{
    for (NvU32 u = 0; u < m_pLayers.size(); ++u)
    {
        if (uParam < m_pLayers[u]->getNTrainableParams())
        {
            m_pLayers[u]->setTrainableParam(uParam, fValue);
            return;
        }
        uParam -= m_pLayers[u]->getNTrainableParams();
    }
    nvAssert(false); // no such param
}
