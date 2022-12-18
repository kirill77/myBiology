#pragma once

#include "activations.h"
#include "layerBatchData.h"

enum LAYER_TYPE { LAYER_TYPE_UNKNOWN = 0, LAYER_TYPE_FCL_IDENTITY, LAYER_TYPE_FCL_MIRRORED };

struct ILayer
{
    // returns input for the next layer
    virtual TensorRef forward(NvU32 uBatch, TensorRef pInput) = 0;
    // returns computed loss for the previous layer
    virtual Tensor *backward(NvU32 uBatch, Tensor &loss,
        float fBiasesLR, float fWeightsLR) = 0;

    virtual void allocateBatchData(NvU32 uBatch, NvU32 n, bool isFirstLayer);

    void saveCurrentStateToBackup()
    {
        m_weightsBackup.copyFrom(m_weights);
        m_biasesBackup.copyFrom(m_biases);
    }
    void restoreStateFromBackup()
    {
        m_weights.copyFrom(m_weightsBackup);
        m_biases.copyFrom(m_biasesBackup);
    }

    void updateLoss(NvU32 uBatch, Tensor& wantedOutput,
        struct LossComputer& lossComputer, Tensor &outLoss, double* pErrorPtr);

    const LAYER_TYPE m_type = LAYER_TYPE_UNKNOWN;

    static std::shared_ptr<ILayer> createLayer(LAYER_TYPE layerType);

    virtual void serialize(ISerializer& s)
    {
        m_weights.serialize("m_weights", s);
        m_biases.serialize("m_biases", s);
        m_weightsBackup.serialize("m_weightsBackup", s);
        m_biasesBackup.serialize("m_biasesBackup", s);
    }

    // functions used to check analytic derivative against numeric ones
    virtual NvU32 getNTrainableParams() const;
    virtual double getTrainableParam(NvU32 uParam);
    virtual void setTrainableParam(NvU32 uParam, double fValue);

protected:
    ILayer(LAYER_TYPE type) : m_type(type)
    {
    }
    std::array<unsigned, 4> m_inputDims = { }, m_outputDims = { };
    Tensor m_weights, m_biases, m_weightsBackup, m_biasesBackup;
    BatchDataContainer m_batchesData;
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
    FullyConnectedLayer() : ILayer(computeFCLType(T_ACTIVATION1, T_ACTIVATION2))
    { }
    virtual void allocateBatchData(NvU32 uBatch, NvU32 n, bool isFirstLayer) override
    {
        __super::allocateBatchData(uBatch, n, isFirstLayer);
        auto& batchData = m_batchesData.accessBatchData(uBatch);

        std::array<unsigned, 4> dimsTmp = m_outputDims;
        dimsTmp[0] = n;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            dimsTmp[1] /= 2;
        }

        nvAssert(batchData.m_beforeActivation == nullptr);
        batchData.m_beforeActivation = std::make_shared<Tensor>(dimsTmp, sizeof(float));
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
        m_weights.init(1, nSummatorsPerOutputCluster, nValuesPerInputCluster, 1, sizeof(float));
        m_weights.clearWithRandomValues<float>(-1, 1, rng);
        m_biases.init(1, nSummatorsPerOutputCluster, 1, 1, sizeof(float));
        m_biases.clearWithRandomValues<float>(-1, 1, rng);

        // and our output will be:
        nvAssert(m_inputDims[0] == m_outputDims[0] && m_inputDims[3] == m_outputDims[3]);
    }

    // returns input for the next layer
    virtual TensorRef forward(NvU32 uBatch, TensorRef pInput) override;
    // returns computed loss for the previous layer
    virtual Tensor *backward(NvU32 uBatch, Tensor& loss,
        float fBiasesLR, float fWeightsLR) override;

    virtual void serialize(ISerializer& s) override
    {
        std::shared_ptr<Indent> pIndent = s.pushIndent("FullyConnectedLayer");
        ILayer::serialize(s);
        s.serializeSimpleType("m_inputDims", m_inputDims);
        s.serializeSimpleType("m_outputDims", m_outputDims);
    }
};
