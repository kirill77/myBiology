#pragma once

#include <array>
#include "basics/serializer.h"
#include "neuralTest.h"
#include "layer.h"

struct NeuralNetwork
{
    NeuralNetwork()
    {
        nvAssert(NeuralTest::isTested());
    }

    virtual NvU32 getNBatches() = 0;
    virtual void initBatch(struct BatchTrainer& batchTrainer, NvU32 uBatch) = 0;

    NvU32 getNLayers() const
    {
        nvAssert(m_pLayers.size() > 0); // derived class must have created layers by that point
        return (NvU32)m_pLayers.size();
    }
    ILayer& getLayer(NvU32 u)
    {
        return *m_pLayers[u];
    }
    void forwardPass(NvU32 uBatch, TensorRef pInput)
    {
        for (NvU32 uLayer = 0; uLayer < m_pLayers.size(); ++uLayer)
        {
            pInput = m_pLayers[uLayer]->forward(uBatch, pInput);
        }
    }
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
    void updateLoss(NvU32 uBatch, Tensor<float>& wantedOutput,
        LossComputer& lossComputer, Tensor<float> &outLoss, float* pErrorPtr)
    {
        (*m_pLayers.rbegin())->updateLoss(uBatch, wantedOutput, lossComputer, outLoss, pErrorPtr);
    }

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

protected:
    std::vector<std::shared_ptr<ILayer>> m_pLayers;
};
