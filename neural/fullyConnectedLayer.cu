﻿#include "neural/tensor.h"
#include "neural/network.h"
#include "myCudaMath.h"

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FCL_Forward
{
    FCL_Forward(Tensor<float>& input, Tensor<float>& output, Tensor<float>& weights, Tensor<float>& biases,
        Tensor<float> &beforeActivation) :
        m_input(input), m_output(output), m_weights(weights), m_biases(biases), m_beforeActivation(beforeActivation)
    {
#if RUN_ON_GPU
        m_input.notifyDeviceBind(false);
        m_output.notifyDeviceBind(true);
        m_weights.notifyDeviceBind(false);
        m_biases.notifyDeviceBind(false);
        m_beforeActivation.notifyDeviceBind(true);
#endif
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
        m_beforeActivation.access(inOutNi, threadY, outWi, inOutCi) = fBeforeActivation;
        float fAfterActivation = TFunction<T_ACTIVATION1>(fBeforeActivation);
        m_output.access(inOutNi, outHi, outWi, inOutCi) = fAfterActivation;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            float fAfterActivation2 = TFunction<T_ACTIVATION2>(fBeforeActivation);
            m_output.access(inOutNi, outHi + 1, outWi, inOutCi) = fAfterActivation2;
        }
    }

    Tensor<float> m_input, m_weights, m_biases, m_beforeActivation, m_output;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
__global__ void fclForwardKernel(FCL_Forward<T_ACTIVATION1, T_ACTIVATION2> p)
{
    p.forward(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
void FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::forward(std::vector<TensorRef>& inputs,
	BatchTrainer &batchTrainer)
{
    NvU32 n = batchTrainer.n();
    LayerBatchData& batchData = batchTrainer.accessLayerData(m_layerId);
    nvAssert(inputs.size() == 1 && batchData.m_outputs.size() == 1); // this layer has one input tensor and one output tensor
    Tensor<float>& input = *inputs[0];
    nvAssert(input.n() == n && m_inputDims[0] == 1 && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor<float>& output = *batchData.m_outputs[0];
    nvAssert(output.n() == n && m_outputDims[0] == 1 && output.h() == m_outputDims[1] && output.w() == m_outputDims[2] && output.c() == m_outputDims[3]);
    Tensor<float>& beforeActivation = *batchData.m_beforeActivation[0];

    dim3 grid(n, m_outputDims[3], 1);
    dim3 block(m_outputDims[2], T_ACTIVATION1 == T_ACTIVATION2 ? m_outputDims[1] : m_outputDims[1] / 2, 1);
    FCL_Forward<T_ACTIVATION1, T_ACTIVATION2> forward(input, output, m_weights, m_biases, beforeActivation);
#if RUN_ON_GPU
    fclForwardKernel << <grid, block >> > (forward);
#else
    for (unsigned iBlockY = 0; iBlockY < grid.y; ++iBlockY)
    {
        for (unsigned iBlockX = 0; iBlockX < grid.x; ++iBlockX)
        {
            for (unsigned iThreadY = 0; iThreadY < block.y; ++iThreadY)
            {
                for (unsigned iThreadX = 0; iThreadX < block.x; ++iThreadX)
                {
                    forward.forward(iBlockX, iBlockY, iThreadX, iThreadY);
                }
            }
        }
    }
#endif
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FCL_Backward
{
    FCL_Backward(float fBiasesLR, float fWeightsLR, Tensor<float>& input, Tensor<float>& weights, Tensor<float>& biases,
    Tensor<float> &deltaInput, Tensor<float> &loss, Tensor<float> &beforeActivation) :
        m_fBiasesLR(fBiasesLR), m_fWeightsLR(fWeightsLR),
        m_input(input), m_weights(weights), m_biases(biases),
        m_deltaInput(deltaInput), m_loss(loss), m_beforeActivation(beforeActivation)
    {
#if RUN_ON_GPU
        m_input.notifyDeviceBind(false);
        m_weights.notifyDeviceBind(true);
        m_biases.notifyDeviceBind(true);
        m_deltaInput.notifyDeviceBind(true);
        m_loss.notifyDeviceBind(false);
        m_beforeActivation.notifyDeviceBind(false);
#endif
    }

    __host__ __device__ void backward(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned outWi = threadX;
        unsigned _outHi = threadY;
        unsigned inWi = blockX;
        unsigned inHi = blockY;

        float fDeltaBias = 0, fDeltaWeight = 0;
        unsigned iWeight = (outWi + _outHi * m_loss.w()) * m_input.h() * m_input.w() + inHi * m_input.w() + inWi;
        float fW = m_weights[iWeight];
        for (unsigned inOutNi = 0; inOutNi < m_loss.n(); ++inOutNi)
        {
            for (unsigned inOutCi = 0; inOutCi < m_loss.c(); ++inOutCi)
            {
                unsigned outHi = _outHi * (T_ACTIVATION1 != T_ACTIVATION2 ? 2 : 1);

                float fLoss[2] = {0 , 0};
                fLoss[0] = m_loss.access(inOutNi, outHi, outWi, inOutCi);
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    fLoss[1] = m_loss.access(inOutNi, outHi + 1, outWi, inOutCi);
                }
                if (fLoss[0] == 0 && (T_ACTIVATION1 == T_ACTIVATION2 || fLoss[1] == 0)) // if no error - nothing to do
                    continue;
                float fBeforeActivation = m_beforeActivation.access(inOutNi, _outHi, outWi, inOutCi);
                float fActivationDer = TFunctionDer<T_ACTIVATION1>(fBeforeActivation);
                float fMult = fLoss[0] * fActivationDer;
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    float fActivation2Der = TFunctionDer<T_ACTIVATION2>(fBeforeActivation);
                    fMult += fLoss[1] * fActivation2Der;
                }
                fDeltaBias += fMult;
                // modify the weight corresponding to this summator
                float fInput = m_input.access(inOutNi, inHi, inWi, inOutCi);
                fDeltaWeight += fMult * fInput;
                if (m_deltaInput.n()) // have we been asked to compute deltaInput?
                {
                    float& fDeltaInput = m_deltaInput.access(inOutNi, inHi, inWi, inOutCi);
                    myAtomicAdd(&fDeltaInput, fMult * fW);
                }
            }
        }
        // bias address only depends on threadId - meaning the same threadIds from different blocks may race
        unsigned iBias = _outHi * m_loss.w() + outWi;
        myAtomicAdd(&m_biases[iBias], fDeltaBias * m_fBiasesLR);
        m_weights[iWeight] += fDeltaWeight * m_fWeightsLR;
    }

    Tensor<float> m_input, m_weights, m_biases, m_deltaInput, m_loss, m_beforeActivation;
    float m_fBiasesLR = 0, m_fWeightsLR = 0;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
__global__ void fclBackwardKernel(FCL_Backward<T_ACTIVATION1, T_ACTIVATION2> backward)
{
    backward.backward(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
void FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::backward(std::vector<TensorRef>& inputs,
    Tensor<float>& loss, float fBiasesLR,
    float fWeightsLR, BatchTrainer &batchTrainer, std::vector<TensorRef>* pDeltaInputs)
{
    NvU32 n = batchTrainer.n();
    LayerBatchData& batchData = batchTrainer.accessLayerData(m_layerId);
    nvAssert(inputs.size() == 1);
    Tensor<float>& input = *inputs[0];
    nvAssert(input.n() == n && m_inputDims[0] == 1 && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor<float> deltaInput;
    if (pDeltaInputs)
    {
        nvAssert(pDeltaInputs->size() == 1);
        deltaInput = *(*pDeltaInputs)[0];
        nvAssert(deltaInput.n() == n && deltaInput.h() == m_inputDims[1] && deltaInput.w() == m_inputDims[2] && deltaInput.c() == m_inputDims[3]);
    }
    nvAssert(loss.n() == n && m_outputDims[0] == 1 && loss.h() == m_outputDims[1] && loss.w() == m_outputDims[2] && loss.c() == m_outputDims[3]);
    if (deltaInput.n())
    {
        deltaInput.clearSubregion(0, (NvU32)deltaInput.size(), EXECUTE_MODE_DEFAULT);
    }
    Tensor<float>& beforeActivation = *batchData.m_beforeActivation[0];
    FCL_Backward<T_ACTIVATION1, T_ACTIVATION2> backward(fBiasesLR, fWeightsLR, input, m_weights, m_biases, deltaInput, loss, beforeActivation);
    nvAssert(T_ACTIVATION1 == T_ACTIVATION2 || loss.h() % 2 == 0);
    unsigned outHiNum = (T_ACTIVATION1 == T_ACTIVATION2 ? loss.h() : loss.h() / 2);
    dim3 grid(input.w(), input.h(), 1);
    dim3 block(loss.w(), outHiNum, 1);
#if RUN_ON_GPU
    fclBackwardKernel << <grid, block >> > (backward);
#else
    input.syncToHost();
    output.syncToHost();
    for (unsigned blockY = 0; blockY < grid.y; ++blockY)
    {
        for (unsigned blockX = 0; blockX < grid.x; ++blockX)
        {
            for (unsigned outHi = 0; outHi < block.y; ++outHi)
            {
                for (unsigned outWi = 0; outWi < block.x; ++outWi)
                {
                    backward.backward(blockX, blockY, outWi, outHi);
                }
            }
        }
    }
#endif
}

// explicit instantiation
template struct FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
template struct FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;

