#pragma once

#include <array>
#include "basics/serializer.h"
#include "layer.h"
#include "batch.h"

struct NeuralNetwork
{
    NeuralNetwork();

    virtual NvU32 getNBatches() = 0;

    virtual NvU32 getNLearningRatesNeeded() const
    {
        return (NvU32)m_pLayers.size();
    }

    std::shared_ptr<Batch> createBatch(NvU32 uBatch)
    {
        std::shared_ptr<Batch> pBatch = createAndInitBatchInternal(uBatch);
        for (NvU32 uLayer = 0; uLayer < m_pLayers.size(); ++uLayer)
        {
            m_pLayers[uLayer]->allocateBatchData(uBatch, pBatch->n(), uLayer == 0);
        }
        return pBatch;
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
    TensorRef updateLoss(NvU32 uBatch, Tensor& wantedOutput,
        LossComputer& lossComputer, double* pErrorPtr)
    {
        TensorRef pLoss = getTmpLossTensor(wantedOutput.getDims());
        (*m_pLayers.rbegin())->updateLoss(uBatch, wantedOutput, lossComputer, *pLoss, pErrorPtr);
        return pLoss;
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

protected:
    virtual std::shared_ptr<Batch> createAndInitBatchInternal(NvU32 uBatch) = 0;
    std::vector<std::shared_ptr<ILayer>> m_pLayers;
    RNGUniform m_rng;

private:
    TensorRef getTmpLossTensor(const std::array<NvU32, 4>& dims)
    {
        return getTmpTensor(m_pTmpLoss, dims);
    }
    TensorRef getTmpTensor(TensorRef& pCache, const std::array<NvU32, 4>& dims);
    TensorRef m_pTmpLoss = nullptr;
    double m_fLRSum = 0;
    int m_nLRSamples = 0;
};
