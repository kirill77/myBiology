#pragma once

#include <array>
#include "basics/vectors.h"
#include "basics/serializer.h"
#include "tensor.h"
#include "l2Computer.h"
#include "activations.h"
#include "neuralTest.h"
#include "batchTrainer.h"
#include "layer.h"

struct NeuralNetwork
{
    NeuralNetwork()
    {
        nvAssert(NeuralTest::isTested());
    }

    virtual NvU32 getNBatches() = 0;
    virtual void initBatch(BatchTrainer& batchTrainer, NvU32 uBatch) = 0;
    double computeAvgLRStats() const
    {
        return (m_nLRSamples == 0) ? 0 : m_fLRSum / m_nLRSamples;
    }
    void resetAvgLRStats()
    {
        m_fLRSum = 0;
        m_nLRSamples = 0;
    }
    NvU32 getNLayers() const
    {
        nvAssert(m_pLayers.size() > 0); // derived class must have created layers by that point
        return (NvU32)m_pLayers.size();
    }
    ILayer& getLayer(NvU32 u)
    {
        return *m_pLayers[u];
    }

protected:
    virtual void serialize(ISerializer& s)
    {
        std::shared_ptr<Indent> pIndent = s.pushIndent("ArrayOfNeuralLayers");
        s.serializeArraySize("m_pLayers", m_pLayers);
        for (NvU32 uLayer = 0; uLayer < m_pLayers.size(); ++uLayer)
        {
            LAYER_TYPE layerType = LAYER_TYPE_UNKNOWN;
            if (m_pLayers[uLayer] != nullptr)
            {
                layerType = m_pLayers[uLayer]->m_type;
            }
            s.serializeSimpleType("layerType", layerType);
            if (layerType == LAYER_TYPE_UNKNOWN)
            {
                nvAssert(m_pLayers[uLayer] == nullptr);
                continue;
            }
            if (m_pLayers[uLayer] == nullptr)
            {
                m_pLayers[uLayer] = ILayer::createLayer(layerType, uLayer);
            }
            char sBuffer[16];
            sprintf_s(sBuffer, "[%d]", uLayer);
            std::shared_ptr<Indent> pIndent = s.pushIndent(sBuffer);
            m_pLayers[uLayer]->serialize(s);
        }
    }

private:
    double m_fLRSum = 0;
    int m_nLRSamples = 0;

protected:
    std::vector<std::shared_ptr<ILayer>> m_pLayers;

public:
    void saveCurrentStateToBackup()
    {
        for (NvU32 u = 0; u < m_pLayers.size(); ++u)
        {
            m_pLayers[u]->saveCurrentStateToBackup();
        }
    }
    void restoreStateFromBackup()
    {
        for (NvU32 u = 0; u < m_pLayers.size(); ++u)
        {
            m_pLayers[u]->restoreStateFromBackup();
        }
    }
    void backwardPass(BatchTrainer& batchTrainer, LossComputer &lossComputer)
    {
        NvU32 uLayer = (NvU32)m_pLayers.size() - 1;
        while (uLayer < m_pLayers.size())
        {
            std::vector<TensorRef>& inputs = batchTrainer.getInputs(uLayer);

            // we don't need to compute deltaInputs for the layer 0
            std::vector<TensorRef>* pDeltaInputs = (uLayer == 0) ? nullptr : &batchTrainer.accessLayerData(uLayer - 1).m_deltaOutputs;
            float fBiasesLR = batchTrainer.getLearningRate(uLayer);
            float fWeightsLR = batchTrainer.getLearningRate(uLayer);
            m_fLRSum += fBiasesLR + fWeightsLR;
            m_nLRSamples += 2;
            if (uLayer == m_pLayers.size() - 1)
            {
                batchTrainer.computeLoss(lossComputer);
                m_pLayers[uLayer]->backward(inputs, batchTrainer.m_loss,
                    fBiasesLR, fWeightsLR, batchTrainer, pDeltaInputs);
            }
            else
            {
                m_pLayers[uLayer]->backward(inputs,
                    *batchTrainer.accessLayerData(uLayer).m_deltaOutputs[0], fBiasesLR, fWeightsLR,
                    batchTrainer, pDeltaInputs);
            }
            --uLayer;
        }
    }
    void forwardPass(BatchTrainer &batchTrainer)
    {
        for (NvU32 uLayer = 0; uLayer < m_pLayers.size(); ++uLayer)
        {
            m_pLayers[uLayer]->forward(batchTrainer.getInputs(uLayer),batchTrainer);
        }
    }
};
