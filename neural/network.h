#pragma once

#include "tensor.h"

template <class ElementType>
struct GPUBuffer
{
    std::vector<ElementType>& beginChanging()
    {
        ++m_hostRev;
        return m_pHost;
    }
    void endChanging()
    {
        ++m_hostRev;
    }
    size_t getNBytes() const { return sizeof(ElementType) * m_pHost.size(); }
    const std::vector<ElementType>& get() const { return m_pHost; }
    ElementType* getDevicePtr();

private:
    NvU32 m_nDeviceElems = 0, m_hostRev = 0, m_deviceRev = 0;
    std::vector<ElementType> m_pHost;
    ElementType* m_pDevice = nullptr;
};

struct ILayer
{

};

struct FullyConnectedLayer
{
    // we assume input layer has dimensions:
    // n - number of clusters
    // width, height - values belonging to the same cluster
    // c = 1
    void init(const Tensor<float> &input, NvU32 outputClusterHeight, NvU32 outputClusterWidth)
    {
        nvAssert(input.c() == 1);
        NvU32 nValuesPerInputCluster = input.h() * input.w();
        NvU32 nValuesPerOutputCluster = outputClusterHeight * outputClusterWidth;
        // fully connected means we have this many weights and biases:
        m_weights.init(1, nValuesPerInputCluster, nValuesPerOutputCluster, 1);
        m_biases.init(1, nValuesPerInputCluster, nValuesPerOutputCluster, 1);
        // and our output will be:
        m_output.init(input.n(), m_weights.h(), m_weights.w(), 1);
    }
    void forward(const Tensor<float>& input);
    // this will also affect m_weights and m_biases
    void backward(const Tensor<float>& deltaOutput, Tensor<float> &deltaInput);

private:
    Tensor<float> m_weights, m_biases, m_output;
};

template <class T>
inline void copy(rtvector<float, 3>& dst, const rtvector<T, 3>& src)
{
    dst[0] = (float)src[0];
    dst[1] = (float)src[1];
    dst[2] = (float)src[2];
}

template <class T>
struct NeuralNetwork
{
    double train(NvU32 nSteps)
    {
        std::vector<Tensor<T>*> pInputs, pOutputs, pDeltaOutputsSum;
        for (NvU32 uStep = 0; uStep < nSteps; ++uStep)
        {
            if (!startBatch_impl(pInputs))
            {
                return 0;
            }
            for (; ; )
            {
                executeTheNetwork(pInputs, pOutputs);
                if (!computeDeltaOutput_impl(pOutputs))
                {
                    nvAssert(false);
                    return 0;
                }
                addToDeltaOutputs(pOutputs, pDeltaOutputsSum);

                if (!continueBatch_impl(pInputs))
                {
                    break;
                }
            }
            backPropagation(pDeltaOutputsSum);
        }
        return computeCurrentError();
    }

    virtual bool startBatch_impl(std::vector<Tensor<T>*> &pInputs) = 0;
    virtual bool continueBatch_impl(std::vector<Tensor<T>*>& pInputs) = 0;
    virtual bool computeDeltaOutput_impl(std::vector<Tensor<T>*>& pOutputs) = 0;

private:
    double computeCurrentError()
    {
        return 5;
    }
    void backPropagation(std::vector<Tensor<T>*>& pDeltaOutputs)
    {
    }
    void executeTheNetwork(std::vector<Tensor<T>*> &pInputs, std::vector<Tensor<T>*> &pOutputs)
    {

    }
    void addToDeltaOutputs(std::vector<Tensor<T>*> &pDeltaOutputs, std::vector<Tensor<T>*> &pDeltaOutputsSum)
    {

    }
    std::vector<ILayer*> m_pLayers;
};
