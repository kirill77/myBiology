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
    virtual void makeSteps(NvU32 nStepsToMake, BatchTrainer& batchTrainer)
    {
        for (NvU32 u = 0; u < nStepsToMake; ++u)
        {
            backwardPass(batchTrainer);
            forwardPass(batchTrainer);
        }
    }
    double getFilteredLearningRate() const { return m_fFilteredLearningRate; }
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
        s.serializeSimpleType("m_fFilteredLearningRate", m_fFilteredLearningRate);
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
    }

private:
    double m_fFilteredLearningRate = 0;

protected:
    std::vector<std::shared_ptr<ILayer>> m_pLayers;

private:
    L2Computer m_l2Computer;

public:
    float computeCurrentError(BatchTrainer &batchTrainer)
    {
        const std::vector<TensorRef>& outputs = batchTrainer.m_pLayerOutputs.rbegin()->m_outputs;
        nvAssert(outputs.size() == batchTrainer.m_wantedOutputs.size());
        for (NvU32 uTensor = 0; uTensor < outputs.size(); ++uTensor)
        {
            Tensor<float>& output = (*outputs[uTensor]);
            Tensor<float>& wantedOutput = (*batchTrainer.m_wantedOutputs[uTensor]);
            m_l2Computer.accumulateL2Error(output, wantedOutput, (uTensor == 0) ? L2_MODE_RESET : L2_MODE_ADD);
        }
        float fError = m_l2Computer.getAccumulatedError();
        return fError;
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
    void backwardPass(BatchTrainer& batchTrainer)
    {
        NvU32 uLayer = (NvU32)m_pLayers.size() - 1;
        while (uLayer < m_pLayers.size())
        {
            std::vector<TensorRef>& _inputs = (uLayer == 0) ?
                batchTrainer.m_inputs : batchTrainer.m_pLayerOutputs[uLayer - 1].m_outputs;

            // we don't need to compute deltaInputs for the layer 0
            std::vector<TensorRef>* pDeltaInputs = (uLayer == 0) ? nullptr : &batchTrainer.m_pLayerOutputs[uLayer - 1].m_deltaOutputs;
            float fBiasesLR = batchTrainer.getLearningRate(uLayer);
            float fWeightsLR = batchTrainer.getLearningRate(uLayer);
            m_fFilteredLearningRate = (fBiasesLR + fWeightsLR) * 0.01 + m_fFilteredLearningRate * 0.99;
            if (uLayer == m_pLayers.size() - 1)
            {
                m_pLayers[uLayer]->backward(_inputs, ILayer::WANTED_OUTPUTS, batchTrainer.m_wantedOutputs,
                    fBiasesLR, fWeightsLR, batchTrainer, pDeltaInputs);
            }
            else
            {
                m_pLayers[uLayer]->backward(_inputs, ILayer::DELTA_OUTPUTS,
                    batchTrainer.m_pLayerOutputs[uLayer].m_deltaOutputs, fBiasesLR, fWeightsLR,
                    batchTrainer, pDeltaInputs);
            }
            --uLayer;
        }
    }
    void forwardPass(BatchTrainer &batchTrainer)
    {
        m_pLayers[0]->forward(batchTrainer.m_inputs, batchTrainer);
        for (NvU32 uLayer = 1; uLayer < m_pLayers.size(); ++uLayer)
        {
            m_pLayers[uLayer]->forward(batchTrainer.m_pLayerOutputs[uLayer - 1].m_outputs,
                batchTrainer);
        }
    }
};
