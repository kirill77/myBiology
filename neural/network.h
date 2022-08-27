#pragma once

#include <array>
#include "basics/vectors.h"
#include "basics/serializer.h"
#include "tensor.h"
#include "l2Computer.h"
#include "activations.h"
#include "neuralTest.h"
#include "batchTrainer.h"

enum LAYER_TYPE { LAYER_TYPE_UNKNOWN = 0, LAYER_TYPE_FCL_IDENTITY, LAYER_TYPE_FCL_MIRRORED };

struct ILayer
{
    virtual void forward(std::vector<TensorRef>& inputs, BatchTrainer& batchTrainer) = 0;

    enum OUTPUTS_DATA_TYPE { WANTED_OUTPUTS, DELTA_OUTPUTS };
    virtual void backward(std::vector<TensorRef>& inputs,
        OUTPUTS_DATA_TYPE outputsDataType, std::vector<TensorRef>& outputsData, float fBiasesLR,
        float fWeightsLR, BatchTrainer& batchTrainer,
        std::vector<TensorRef>* pDeltaInputs = nullptr) = 0;

    virtual void allocateBatchData(BatchTrainer& batchTrainer)
    {
        auto &batchData = batchTrainer.m_pLayerOutputs[m_layerId];

        std::vector<TensorRef>& deltaOutputs = batchData.m_deltaOutputs;
        deltaOutputs.resize(1);
        if (deltaOutputs[0] == nullptr)
        {
            deltaOutputs[0] = std::make_shared<Tensor<float>>();
        }
        deltaOutputs[0]->init(m_outputDims);

        std::vector<TensorRef>& outputs = batchData.m_outputs;
        outputs.resize(1);
        if (outputs[0] == nullptr)
        {
            outputs[0] = std::make_shared<Tensor<float>>();
        }
        outputs[0]->init(m_outputDims);
    }
    void saveCurrentStateToBackup()
    {
        m_weightsBackup.resize(m_weights.size());
        m_weightsBackup.copySubregionFrom(0, m_weights, 0, (NvU32)m_weights.size());
        m_biasesBackup.resize(m_biases.size());
        m_biasesBackup.copySubregionFrom(0, m_biases, 0, (NvU32)m_biases.size());
    }
    void restoreStateFromBackup()
    {
        m_weights.copySubregionFrom(0, m_weightsBackup, 0, (NvU32)m_weightsBackup.size());
        m_biases.copySubregionFrom(0, m_biasesBackup, 0, (NvU32)m_biasesBackup.size());
    }

    Tensor<float> m_weights, m_biases, m_weightsBackup, m_biasesBackup;
    const LAYER_TYPE m_type = LAYER_TYPE_UNKNOWN;

    static std::shared_ptr<ILayer> createLayer(LAYER_TYPE layerType, NvU32 layerId);

    virtual void serialize(ISerializer& s)
    {
        m_weights.serialize("m_weights", s);
        m_biases.serialize("m_biases", s);
        m_weightsBackup.serialize("m_weightsBackup", s);
        m_biasesBackup.serialize("m_biasesBackup", s);
    }

    const NvU32 m_layerId = 0; // layer index unique for inside the same neural network

protected:
    ILayer(LAYER_TYPE type, NvU32 layerId) : m_type(type), m_layerId(layerId)
    {
    }
    std::array<unsigned, 4> m_inputDims = { }, m_outputDims = { };
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FullyConnectedLayer : public ILayer
{
    static LAYER_TYPE computeFCLType(ACTIVATION a1, ACTIVATION a2)
    {
        if (a1 == ACTIVATION_IDENTITY && a2 == ACTIVATION_IDENTITY)
        {
            return LAYER_TYPE_FCL_IDENTITY;
        }
        if (a1 == ACTIVATION_RELU && a2 == ACTIVATION_MRELU)
        {
            return LAYER_TYPE_FCL_MIRRORED;
        }
        nvAssert(false);
        return LAYER_TYPE_UNKNOWN;
    }
    FullyConnectedLayer(NvU32 layerId) : ILayer(computeFCLType(T_ACTIVATION1, T_ACTIVATION2),
        layerId)
    { }
    virtual void allocateBatchData(BatchTrainer& batchTrainer) override
    {
        std::array<unsigned, 4> dimsTmp = m_outputDims;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            dimsTmp[1] /= 2;
        }
        std::vector<TensorRef>& ba = batchTrainer.m_pLayerOutputs[m_layerId].m_beforeActivation;
        ba.resize(1);
        if (ba[0] == nullptr)
        {
            ba[0] = std::make_shared<Tensor<float>>();
        }
        ba[0]->init(dimsTmp);
        __super::allocateBatchData(batchTrainer);
    }
    void init(const std::array<unsigned, 4> &inputDims, const std::array<unsigned, 4> &outputDims, BatchTrainer &batchTrainer)
    {
        // upper half of neurons uses different activation function 
        m_inputDims = inputDims;
        m_outputDims = outputDims;

        NvU32 nValuesPerInputCluster = m_inputDims[1] * m_inputDims[2];
        NvU32 nSummatorsPerOutputCluster = m_outputDims[1] * m_outputDims[2];

        // if activations are different, then each each summator is duplicated to two outputs (each output will have its own activation function)
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            nvAssert(outputDims[1] % 2 == 0);
            nSummatorsPerOutputCluster /= 2;
        }

        // fully connected means we have this many weights and biases:
        m_weights.init(1, nSummatorsPerOutputCluster, nValuesPerInputCluster, 1);
        m_weights.clearWithRandomValues(-1, 1);
        m_biases.init(1, nSummatorsPerOutputCluster, 1, 1);
        m_biases.clearWithRandomValues(-1, 1);

        // and our output will be:
        nvAssert(m_inputDims[0] == m_outputDims[0] && m_inputDims[3] == m_outputDims[3]);
    }
    virtual void forward(std::vector<TensorRef>& inputs, BatchTrainer &batchTrainer) override;
    virtual void backward(std::vector<TensorRef>& inputs,
        OUTPUTS_DATA_TYPE outputsDataType, std::vector<TensorRef>& outputsData, float fBiasesLR,
        float fWeightsLR, BatchTrainer &batchTrainer,
        std::vector<TensorRef>* pDeltaInputs = nullptr) override;

    virtual void serialize(ISerializer& s) override
    {
        std::shared_ptr<Indent> pIndent = s.pushIndent("FullyConnectedLayer");
        ILayer::serialize(s);
        s.serializeSimpleType("m_inputDims", m_inputDims);
        s.serializeSimpleType("m_outputDims", m_outputDims);
    }
};

struct NeuralNetwork
{
    NeuralNetwork()
    {
        nvAssert(NeuralTest::isTested());
    }

    void initBatch(std::vector<TensorRef>& inputs, std::vector<TensorRef>& wantedOutputs, BatchTrainer& batchTrainer)
    {
        if (m_pLayers.size() == 0)
        {
            createLayers_impl(m_pLayers, batchTrainer);
        }
        nvAssert(m_pLayers.size() != 0);

        batchTrainer.init(m_pLayers, (NvU32)m_pLayers.size(), *this, inputs, wantedOutputs);
    }
    virtual void makeSteps(NvU32 nStepsToMake, BatchTrainer& batchTrainer)
    {
        for (NvU32 u = 0; u < nStepsToMake; ++u)
        {
            backwardPass(batchTrainer);
            forwardPass(batchTrainer);
        }
    }

    double getFilteredLearningRate() const { return m_fFilteredLearningRate; }

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

    virtual bool createLayers_impl(std::vector<std::shared_ptr<ILayer>> &pLayers,
        BatchTrainer& batchTrainer) = 0;
    std::vector<std::shared_ptr<ILayer>> m_pLayers;
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
