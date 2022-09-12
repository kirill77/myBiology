#pragma once

#include "activations.h"
#include "layerBatchData.h"

enum LAYER_TYPE { LAYER_TYPE_UNKNOWN = 0, LAYER_TYPE_FCL_IDENTITY, LAYER_TYPE_FCL_MIRRORED };

struct ILayer
{
    virtual void forward(TensorRef pInput, LayerBatchData& data, NvU32 n) = 0;

    virtual void backward(TensorRef pInput,
        Tensor<float> &loss, float fBiasesLR,
        float fWeightsLR, LayerBatchData& data, NvU32 n,
        Tensor<float> *pPrevLoss = nullptr) = 0;

    virtual void allocateBatchData(NvU32 uBatch, NvU32 n);

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

    void updateLoss(NvU32 uBatch, Tensor<float>& wantedOutput, struct LossComputer& lossComputer, float* pErrorPtr);

    const LAYER_TYPE m_type = LAYER_TYPE_UNKNOWN;
    const NvU32 m_layerId = 0; // layer index unique for inside the same neural network

    static std::shared_ptr<ILayer> createLayer(LAYER_TYPE layerType, NvU32 layerId);

    virtual void serialize(ISerializer& s)
    {
        m_weights.serialize("m_weights", s);
        m_biases.serialize("m_biases", s);
        m_weightsBackup.serialize("m_weightsBackup", s);
        m_biasesBackup.serialize("m_biasesBackup", s);
    }

    std::vector<LayerBatchData> m_batches;

protected:
    ILayer(LAYER_TYPE type, NvU32 layerId) : m_type(type), m_layerId(layerId)
    {
    }
    std::array<unsigned, 4> m_inputDims = { }, m_outputDims = { };
    Tensor<float> m_weights, m_biases, m_weightsBackup, m_biasesBackup;
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
    virtual void allocateBatchData(NvU32 uBatch, NvU32 n) override
    {
        __super::allocateBatchData(uBatch, n);
        auto& batchData = m_batches[uBatch];

        std::array<unsigned, 4> dimsTmp = m_outputDims;
        dimsTmp[0] = n;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            dimsTmp[1] /= 2;
        }
        std::vector<TensorRef>& ba = batchData.m_beforeActivation;
        ba.resize(1);
        if (ba[0] == nullptr)
        {
            ba[0] = std::make_shared<Tensor<float>>();
        }
        ba[0]->init(dimsTmp);
    }
    void init(const std::array<unsigned, 4> &inputDims, const std::array<unsigned, 4> &outputDims)
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
        RNGUniform rng;
        m_weights.init(1, nSummatorsPerOutputCluster, nValuesPerInputCluster, 1);
        m_weights.clearWithRandomValues(-1, 1, rng);
        m_biases.init(1, nSummatorsPerOutputCluster, 1, 1);
        m_biases.clearWithRandomValues(-1, 1, rng);

        // and our output will be:
        nvAssert(m_inputDims[0] == m_outputDims[0] && m_inputDims[3] == m_outputDims[3]);
    }
    virtual void forward(TensorRef pInput, LayerBatchData &data, NvU32 n) override;
    virtual void backward(TensorRef pInput,
        Tensor<float>& loss, float fBiasesLR,
        float fWeightsLR, LayerBatchData &data, NvU32 n,
        Tensor<float> *pPrevLoss = nullptr) override;

    virtual void serialize(ISerializer& s) override
    {
        std::shared_ptr<Indent> pIndent = s.pushIndent("FullyConnectedLayer");
        ILayer::serialize(s);
        s.serializeSimpleType("m_inputDims", m_inputDims);
        s.serializeSimpleType("m_outputDims", m_outputDims);
    }
};
