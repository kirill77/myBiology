#pragma once

#include <memory>
#include <array>
#include <vector>
#include "basics/vectors.h"
#include "tensor.h"
#include "l2Computer.h"
#include "activations.h"
#include "neuralTest.h"

typedef std::shared_ptr<Tensor<float>> TensorRef;

struct ILayer
{
    virtual void forward(std::vector<TensorRef>& inputs) = 0;

    enum OUTPUTS_DATA_TYPE { WANTED_OUTPUTS, DELTA_OUTPUTS };
    virtual void backward(std::vector<TensorRef>& inputs,
        OUTPUTS_DATA_TYPE outputsDataType, std::vector<TensorRef>& outputsData, float fLearningRate, std::vector<TensorRef>* pDeltaInputs = nullptr) = 0;

    void allocateDeltaOutputs()
    {
        m_deltaOutputs.resize(m_outputs.size());
        for (NvU32 uOutput = 0; uOutput < m_deltaOutputs.size(); ++uOutput)
        {
            if (m_deltaOutputs[uOutput] == nullptr)
            {
                m_deltaOutputs[uOutput] = std::make_shared<Tensor<float>>();
            }
            m_deltaOutputs[uOutput]->init(m_outputs[uOutput]->getDims());
        }
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

    std::vector<TensorRef> m_outputs, m_deltaOutputs;
    Tensor<float> m_weights, m_biases, m_weightsBackup, m_biasesBackup, m_beforeActivation;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FullyConnectedLayer : public ILayer
{
    FullyConnectedLayer(const std::array<unsigned, 4> &inputDims, const std::array<unsigned, 4> &outputDims)
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
        TensorRef output = std::make_shared<Tensor<float>>();
        output->init(outputDims);
        m_outputs.push_back(output);

        {
            std::array<unsigned, 4> dimsTmp = outputDims;
            if (T_ACTIVATION1 != T_ACTIVATION2)
            {
                dimsTmp[1] /= 2;
            }
            m_beforeActivation.init(dimsTmp);
        }
    }
    virtual void forward(std::vector<TensorRef>& inputs) override;
    virtual void backward(std::vector<TensorRef>& inputs,
        OUTPUTS_DATA_TYPE outputsDataType, std::vector<TensorRef>& outputsData, float fLearningRate, std::vector<TensorRef>* pDeltaInputs = nullptr) override;

private:
    std::array<unsigned, 4> m_inputDims, m_outputDims;
};

template <class T>
inline void copy(rtvector<float, 3>& dst, const rtvector<T, 3>& src)
{
    dst[0] = (float)src[0];
    dst[1] = (float)src[1];
    dst[2] = (float)src[2];
}

struct NeuralNetwork
{
    NeuralNetwork()
    {
        nvAssert(NeuralTest::isTested());
    }
    double train(NvU32 nStepsToMake, std::vector<TensorRef> &inputs, std::vector<TensorRef> &wantedOutputs)
    {
        NvU32 uEndStep = m_nCompletedTrainSteps + nStepsToMake;
        if (m_pLayers.size() == 0)
        {
            createLayers_impl(m_pLayers);
            nvAssert(m_pLayers.size() != 0);
            // allocate delta outputs
            for (NvU32 uLayer = 0; uLayer < m_pLayers.size() - 1; ++uLayer)
            {
                m_pLayers[uLayer]->allocateDeltaOutputs();
            }
        }

        if (m_nCompletedTrainSteps == 0)
        {
            forwardPass(inputs);
            m_fLastError = computeCurrentError(wantedOutputs);
            nvAssert(isfinite(m_fLastError));
            ++m_nCompletedTrainSteps;
            saveCurrentStateToBackup();
        }

        for ( ; m_nCompletedTrainSteps < uEndStep; ++m_nCompletedTrainSteps)
        {
            backwardPass(inputs, wantedOutputs);
            forwardPass(inputs);

            if (++m_nStepsWithoutErrorCheck >= m_nStepsPerErrorCheck)
            {
                m_nStepsWithoutErrorCheck = 0;
                float fCurrentError = computeCurrentError(wantedOutputs);
                if (!isfinite(fCurrentError) || fCurrentError > m_fLastError)
                {
                    restoreStateFromBackup();
                    m_nStepsPerErrorCheck = 1;
                    m_fLearningRate /= 2;
                    forwardPass(inputs);
                }
                else
                {
                    m_fLastError = fCurrentError;
                    saveCurrentStateToBackup();
                    m_nStepsPerErrorCheck = std::min(m_nStepsPerErrorCheck * 2, 128U);
                }
            }
        }

        return m_fLastError;
    }
    double getLastError() const { return m_fLastError; }
    NvU32 getNCompletedTrainSteps() const { return m_nCompletedTrainSteps; }
    float getLearningRate() const { return m_fLearningRate; }

private:
    float m_fLearningRate = 1, m_fLastError = -1;
    NvU32 m_nCompletedTrainSteps = 0, m_nStepsPerErrorCheck = 1, m_nStepsWithoutErrorCheck = 0;

    virtual bool createLayers_impl(std::vector<std::shared_ptr<ILayer>> &pLayers) = 0;
    std::vector<std::shared_ptr<ILayer>> m_pLayers;
    L2Computer m_l2Computer;

    float computeCurrentError(std::vector<TensorRef>& wantedOutputs)
    {
        const std::vector<TensorRef>& outputs = (*m_pLayers.rbegin())->m_outputs;
        nvAssert(outputs.size() == wantedOutputs.size());
        for (NvU32 uTensor = 0; uTensor < outputs.size(); ++uTensor)
        {
            Tensor<float>& output = (*outputs[uTensor]);
            Tensor<float>& wantedOutput = (*wantedOutputs[uTensor]);
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
    void backwardPass(std::vector<TensorRef>& inputs, std::vector<TensorRef>& wantedOutputs)
    {
        NvU32 uLayer = (NvU32)m_pLayers.size() - 1;
        while (uLayer < m_pLayers.size())
        {
            std::vector<TensorRef>& _inputs = (uLayer == 0) ? inputs : m_pLayers[uLayer - 1]->m_outputs;

            // we don't need to compute deltaInputs for the layer 0
            std::vector<TensorRef>* pDeltaInputs = (uLayer == 0) ? nullptr : &m_pLayers[uLayer - 1]->m_deltaOutputs;
            if (uLayer == m_pLayers.size() - 1)
            {
                m_pLayers[uLayer]->backward(_inputs, ILayer::WANTED_OUTPUTS, wantedOutputs, m_fLearningRate, pDeltaInputs);
            }
            else
            {
                m_pLayers[uLayer]->backward(_inputs, ILayer::DELTA_OUTPUTS, m_pLayers[uLayer]->m_deltaOutputs, m_fLearningRate, pDeltaInputs);
            }
            --uLayer;
        }
    }
    void forwardPass(std::vector<TensorRef>& pInputs)
    {
        m_pLayers[0]->forward(pInputs);
        for (NvU32 uLayer = 1; uLayer < m_pLayers.size(); ++uLayer)
        {
            m_pLayers[uLayer]->forward(m_pLayers[uLayer - 1]->m_outputs);
        }
    }
};
