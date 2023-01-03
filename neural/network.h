#pragma once

#include <array>
#include "basics/serializer.h"
#include "layer.h"
#include "batch.h"

struct NeuralNetwork
{
    NeuralNetwork();

    virtual NvU32 getNLearningRatesNeeded() const
    {
        return (NvU32)m_pLayers.size();
    }
    
    void allocateBatchData(NvU32 uBatch, NvU32 batchN)
    {
        for (NvU32 uLayer = 0; uLayer < m_pLayers.size(); ++uLayer)
        {
            m_pLayers[uLayer]->allocateBatchData(uBatch, batchN, uLayer == 0);
        }
    } 

    // returns network output tensor
    TensorRef forwardPass(NvU32 uBatch, TensorRef pInput)
    {
        for (NvU32 uLayer = 0; uLayer < m_pLayers.size(); ++uLayer)
        {
            pInput = m_pLayers[uLayer]->forward(uBatch, pInput);
        }
        return pInput;
    }
    void backwardPass(NvU32 uBatch, Tensor* pLoss, struct LearningRates& lr);
    void saveCurrentStateToBackup()
    {
        for (NvU32 u = 0; u < m_pLayers.size(); ++u)
        {
            m_pLayers[u]->saveCurrentStateToBackup();
        }
    }
    void restoreStateFromBackup(CopyType copyType = ShallowCopy)
    {
        for (NvU32 u = 0; u < m_pLayers.size(); ++u)
        {
            m_pLayers[u]->restoreStateFromBackup(copyType);
        }
    }
    void updateLoss(NvU32 uBatch, Tensor& wantedOutput,
        LossComputer& lossComputer, TensorRef pOutLoss, double* pErrorPtr)
    {
        (*m_pLayers.rbegin())->updateLoss(uBatch, wantedOutput, lossComputer, *pOutLoss, pErrorPtr);
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
                m_pLayers[uLayer] = ILayer::createLayer(layerType);
            }
            char sBuffer[16];
            sprintf_s(sBuffer, "[%d]", uLayer);
            std::shared_ptr<Indent> pIndent = s.pushIndent(sBuffer);
            m_pLayers[uLayer]->serialize(s);
        }
    }
    double computeAvgLRStats() const
    {
        return (m_nLRSamples == 0) ? 0 : m_fLRSum / m_nLRSamples;
    }
    void resetAvgLRStats()
    {
        m_fLRSum = 0;
        m_nLRSamples = 0;
    }

    virtual NvU32 getNTrainableParams() const;
    virtual double getTrainableParam(NvU32 uParam);
    virtual void setTrainableParam(NvU32 uParam, double fValue);

    virtual std::shared_ptr<NeuralNetwork> cloneToPrecision(NvU32 elemSize)
    {
        std::shared_ptr<NeuralNetwork> p = std::make_shared<NeuralNetwork>(*this);
        for (NvU32 u = 0; u < m_pLayers.size(); ++u)
        {
            p->m_pLayers[u] = m_pLayers[u]->cloneToPrecision(elemSize);
        }
        return p;
    }

    void addLayer(std::shared_ptr<ILayer> pLayer)
    {
        m_pLayers.push_back(pLayer);
    }

protected:
    std::vector<std::shared_ptr<ILayer>> m_pLayers;

private:
    double m_fLRSum = 0;
    int m_nLRSamples = 0;
};

struct DataLoader
{
    virtual void serialize(ISerializer& s)
    {
    }
    virtual std::shared_ptr<NeuralNetwork> createNetwork() = 0;
    virtual NvU32 getNBatches() = 0;
    virtual std::shared_ptr<Batch> createBatch(NvU32 uBatch) = 0;
};
