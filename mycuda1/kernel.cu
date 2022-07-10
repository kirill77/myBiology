﻿#include "neural/tensor.h"
#include "neural/network.h"

#define RUN_ON_CPU 1

// when we bind buffer for device access, we have to make sure GPU memory is all up-to-date
template <class T>
void GPUBuffer<T>::notifyDeviceBind(bool isWriteBind)
{
    if (this != m_pOrig)
    {
        m_pOrig->notifyDeviceBind(isWriteBind);
        return;
    }
    if (m_hostRev < m_deviceRev)
        return;
    if (m_hostRev > m_deviceRev)
    {
        if (m_nDeviceElems != m_nHostElems)
        {
            if (m_pDevice)
            {
                cudaFree(m_pDevice);
            }
            if (m_nHostElems == 0)
            {
                m_pDevice = nullptr;
            }
            else
            {
                cudaMalloc(&m_pDevice, m_nHostElems * sizeof(T));
            }
            m_nDeviceElems = m_nHostElems;
        }
        cudaMemcpy(m_pDevice, m_pHost, m_nHostElems * sizeof(T), cudaMemcpyHostToDevice);
    }
    m_deviceRev = m_hostRev + (isWriteBind ? 1 : 0);
}
template <class T>
void GPUBuffer<T>::syncToHost()
{
    if (this != m_pOrig)
    {
        m_pOrig->syncToHost();
        return;
    }
    if (m_hostRev >= m_deviceRev)
        return;
    nvAssert(m_nHostElems == m_nDeviceElems);
    cudaMemcpy(m_pHost, m_pDevice, m_nHostElems * sizeof(T), cudaMemcpyDeviceToHost);
    m_hostRev = m_deviceRev;
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FullyConnectedLayerCuda
{
    FullyConnectedLayerCuda(Tensor<float>& input, Tensor<float>& output, Tensor<float>& weights, Tensor<float>& biases) :
        m_input(input), m_output(output), m_weights(weights), m_biases(biases)
    {
    }

    __host__ __device__ void forward(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned inOutNi = blockX;
        unsigned inOutCi = blockY;
        unsigned outWi = threadX;
        unsigned outHi = (T_ACTIVATION1 != T_ACTIVATION2) ? threadY * 2 : threadY;

        unsigned iBias = outHi / (T_ACTIVATION1 == T_ACTIVATION2 ? 1 : 2) * m_output.w() + outWi;
        unsigned iWeight = m_input.h() * m_input.w() * iBias;
        float fBeforeActivation = m_biases[iBias];
        for (unsigned inHi = 0; inHi < m_input.h(); ++inHi)
        {
            for (unsigned inWi = 0; inWi < m_input.w(); ++inWi)
            {
                fBeforeActivation += m_input.access(inOutNi, inHi, inWi, inOutCi) * m_weights[iWeight++];
            }
        }
        float fAfterActivation = TFunction<T_ACTIVATION1>(fBeforeActivation);
        m_output.access(inOutNi, outHi, outWi, inOutCi) = fAfterActivation;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            float fAfterActivation2 = TFunction<T_ACTIVATION2>(fBeforeActivation);
            m_output.access(inOutNi, outHi + 1, outWi, inOutCi) = fAfterActivation2;
        }
    }

    Tensor<float> m_input, m_weights, m_biases, m_output;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
__global__ void fullyConnectedLayerForward(FullyConnectedLayerCuda<T_ACTIVATION1, T_ACTIVATION2> p)
{
    p.forward(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
void FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::forward(std::vector<TensorRef>& inputs)
{
    nvAssert(inputs.size() == 1 && m_outputs.size() == 1); // this layer has one input tensor and one output tensor
    Tensor<float>& input = *inputs[0];
    nvAssert(input.n() == m_inputDims[0] && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor<float>& output = *m_outputs[0];
    nvAssert(output.n() == m_outputDims[0] && output.h() == m_outputDims[1] && output.w() == m_outputDims[2] && output.c() == m_outputDims[3]);

    dim3 grid(m_outputDims[0], m_outputDims[3], 1);
    dim3 block(m_outputDims[2], T_ACTIVATION1 == T_ACTIVATION2 ? m_outputDims[1] : m_outputDims[1] / 2, 1);
#if !RUN_ON_CPU
    input.notifyDeviceBind(false);
    output.notifyDeviceBind(true);
    m_weights.notifyDeviceBind(false);
    m_biases.notifyDeviceBind(false);
#endif
    FullyConnectedLayerCuda<T_ACTIVATION1, T_ACTIVATION2> cudaLayer(input, output, m_weights, m_biases);
#if RUN_ON_CPU
    for (unsigned iBlockY = 0; iBlockY < grid.y; ++iBlockY)
    {
        for (unsigned iBlockX = 0; iBlockX < grid.x; ++iBlockX)
        {
            for (unsigned iThreadY = 0; iThreadY < block.y; ++iThreadY)
            {
                for (unsigned iThreadX = 0; iThreadX < block.x; ++iThreadX)
                {
                    cudaLayer.forward(iBlockX, iBlockY, iThreadX, iThreadY);
                }
            }
        }
    }
#else
    fullyConnectedLayerForward << <grid, block >> > (cudaLayer);
#endif
}

#if 0
template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct Backward
{
    Backward(Tensor<float>& input, Tensor<float>& output, Tensor<float>& weights, Tensor<float>& biases) :
        m_input(input), m_output(output), m_weights(weights), m_biases(biases)
    {
    }

    __host__ __device__ void go(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned outHi = threadY * (T_ACTIVATION1 == T_ACTIVATION2 ? 1 : 2);
        unsigned iWeight = (outWi + _outHi * m_output.w()) * m_input.h() * m_input.w();
        unsigned iBias = _outHi * m_output.w() + outWi;

        std::array<float, 2> fWantedDeltaOut = { };
        fWantedDeltaOut[0] = outputData.access(inOutNi, outHi, outWi, inOutCi);
        if (outputsDataType == WANTED_OUTPUTS)
        {
            fWantedDeltaOut[0] -= m_outputs[0]->access(inOutNi, outHi, outWi, inOutCi);
        }
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            fWantedDeltaOut[1] = outputData.access(inOutNi, outHi + 1, outWi, inOutCi);
            if (outputsDataType == WANTED_OUTPUTS)
            {
                fWantedDeltaOut[1] -= m_outputs[0]->access(inOutNi, outHi + 1, outWi, inOutCi);
            }
        }
        if (fWantedDeltaOut[0] == 0 && (T_ACTIVATION1 == T_ACTIVATION2 || fWantedDeltaOut[1] == 0)) // if no error - nothing to do
            continue;
        float fBeforeActivation = m_biases[iBias];
        for (unsigned inHi = 0, iiWeight = iWeight; inHi < m_inputDims[1]; ++inHi)
        {
            for (unsigned inWi = 0; inWi < m_inputDims[2]; ++inWi, ++iiWeight)
            {
                fBeforeActivation += input.access(inOutNi, inHi, inWi, inOutCi) * m_weights[iiWeight];
            }
        }
        float fActivationDer = TFunctionDer<T_ACTIVATION1>(fBeforeActivation);
        float fMult = fWantedDeltaOut[0] * fActivationDer;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            float fActivation2Der = TFunctionDer<T_ACTIVATION2>(fBeforeActivation);
            fMult += fWantedDeltaOut[1] * fActivation2Der;
        }
        fMult *= fLearningRate;
        // modify the bias
        m_biases[iBias] += fMult;
        // modify all the weights corresponding to this summator
        for (unsigned inHi = 0, iiWeight = iWeight; inHi < m_inputDims[1]; ++inHi)
        {
            for (unsigned inWi = 0; inWi < m_inputDims[2]; ++inWi, ++iiWeight)
            {
                float fInput = input.access(inOutNi, inHi, inWi, inOutCi);
                float fW = m_weights[iiWeight];
                m_weights[iiWeight] += fMult * fInput;
                if (pDeltaInput) // have we been asked to compute deltaInput?
                {
                    float& fDeltaInput = pDeltaInput->access(inOutNi, inHi, inWi, inOutCi);
                    fDeltaInput += fMult * (fW + m_weights[iiWeight]) / 2;
                }
            }
        }
    }

    Tensor<float> m_input, m_weights, m_biases, m_deltaOutput, m_output, m_wantedOutput;
};
#endif

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
void FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::backward(std::vector<TensorRef>& inputs,
    OUTPUTS_DATA_TYPE outputsDataType, std::vector<TensorRef>& outputsData, float fLearningRate, std::vector<TensorRef>* pDeltaInputs = nullptr)
{
    nvAssert(inputs.size() == 1);
    Tensor<float>& input = *inputs[0];
    input.syncToHost();
    nvAssert(input.n() == m_inputDims[0] && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor<float> deltaInput;
    if (pDeltaInputs)
    {
        nvAssert(pDeltaInputs->size() == 1);
        deltaInput = *(*pDeltaInputs)[0];
        nvAssert(deltaInput.n() == m_inputDims[0] && deltaInput.h() == m_inputDims[1] && deltaInput.w() == m_inputDims[2] && deltaInput.c() == m_inputDims[3]);
    }
    nvAssert(outputsData.size() == 1);
    Tensor<float> wantedOutput = *outputsData[0];
    Tensor<float> output = *m_outputs[0];
    m_outputs[0]->syncToHost();
    nvAssert(wantedOutput.n() == m_outputDims[0] && wantedOutput.h() == m_outputDims[1] && wantedOutput.w() == m_outputDims[2] && wantedOutput.c() == m_outputDims[3]);
    for (unsigned inOutNi = 0; inOutNi < m_outputDims[0]; ++inOutNi)
    {
        for (unsigned inOutCi = 0; inOutCi < m_outputDims[3]; ++inOutCi)
        {
            if (deltaInput.n())
            {
                // clear delta input
                for (unsigned inHi = 0; inHi < m_inputDims[1]; ++inHi)
                {
                    for (unsigned inWi = 0; inWi < m_inputDims[2]; ++inWi)
                    {
                        deltaInput.access(inOutNi, inHi, inWi, inOutCi) = 0;
                    }
                }
            }
            unsigned _outHiNum = (T_ACTIVATION1 == T_ACTIVATION2 ? m_outputDims[1] : m_outputDims[1] / 2);
            for (unsigned _outHi = 0; _outHi < _outHiNum; ++_outHi)
            {
                for (unsigned outWi = 0; outWi < m_outputDims[2]; ++outWi)
                {
                    unsigned outHi = _outHi * (T_ACTIVATION1 == T_ACTIVATION2 ? 1 : 2);
                    unsigned iWeight = (outWi + _outHi * m_outputDims[2]) * m_inputDims[1] * m_inputDims[2];
                    unsigned iBias = _outHi * m_outputDims[2] + outWi;

                    std::array<float, 2> fWantedDeltaOut = { };
                    fWantedDeltaOut[0] = wantedOutput.access(inOutNi, outHi, outWi, inOutCi);
                    if (outputsDataType == WANTED_OUTPUTS)
                    {
                        fWantedDeltaOut[0] -= output.access(inOutNi, outHi, outWi, inOutCi);
                    }
                    if (T_ACTIVATION1 != T_ACTIVATION2)
                    {
                        fWantedDeltaOut[1] = wantedOutput.access(inOutNi, outHi + 1, outWi, inOutCi);
                        if (outputsDataType == WANTED_OUTPUTS)
                        {
                            fWantedDeltaOut[1] -= output.access(inOutNi, outHi + 1, outWi, inOutCi);
                        }
                    }
                    if (fWantedDeltaOut[0] == 0 && (T_ACTIVATION1 == T_ACTIVATION2 || fWantedDeltaOut[1] == 0)) // if no error - nothing to do
                        continue;
                    float fBeforeActivation = m_biases[iBias];
                    for (unsigned inHi = 0, iiWeight = iWeight; inHi < m_inputDims[1]; ++inHi)
                    {
                        for (unsigned inWi = 0; inWi < m_inputDims[2]; ++inWi, ++iiWeight)
                        {
                            fBeforeActivation += input.access(inOutNi, inHi, inWi, inOutCi) * m_weights[iiWeight];
                        }
                    }
                    float fActivationDer = TFunctionDer<T_ACTIVATION1>(fBeforeActivation);
                    float fMult = fWantedDeltaOut[0] * fActivationDer;
                    if (T_ACTIVATION1 != T_ACTIVATION2)
                    {
                        float fActivation2Der = TFunctionDer<T_ACTIVATION2>(fBeforeActivation);
                        fMult += fWantedDeltaOut[1] * fActivation2Der;
                    }
                    fMult *= fLearningRate;
                    // modify the bias
                    m_biases[iBias] += fMult;
                    // modify all the weights corresponding to this summator
                    for (unsigned inHi = 0, iiWeight = iWeight; inHi < m_inputDims[1]; ++inHi)
                    {
                        for (unsigned inWi = 0; inWi < m_inputDims[2]; ++inWi, ++iiWeight)
                        {
                            float fInput = input.access(inOutNi, inHi, inWi, inOutCi);
                            float fW = m_weights[iiWeight];
                            m_weights[iiWeight] += fMult * fInput;
                            if (deltaInput.n()) // have we been asked to compute deltaInput?
                            {
                                float& fDeltaInput = deltaInput.access(inOutNi, inHi, inWi, inOutCi);
                                fDeltaInput += fMult * (fW + m_weights[iiWeight]) / 2;
                            }
                        }
                    }
                }
            }
        }
    }
}

template struct FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
template struct FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
template struct GPUBuffer<float>;