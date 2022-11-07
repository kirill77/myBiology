#include "neural/tensor.h"
#include "neural/network.h"
#include "myCudaMath.h"

#if RUN_ON_GPU
bool g_bExecuteOnTheGPU = true;
#else
bool g_bExecuteOnTheGPU = false;
#endif

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FCL_Forward
{
    FCL_Forward(Tensor<float>& input, Tensor<float>& output, Tensor<float>& weights, Tensor<float>& biases,
        Tensor<float> &beforeActivation) :
        m_input(input), m_output(output), m_weights(weights), m_biases(biases), m_beforeActivation(beforeActivation)
    {
        if (g_bExecuteOnTheGPU)
        {
            m_input.notifyDeviceBind(false);
            m_output.notifyDeviceBind(true);
            m_weights.notifyDeviceBind(false);
            m_biases.notifyDeviceBind(false);
            m_beforeActivation.notifyDeviceBind(true);
        }
        else
        {
            m_input.syncToHost();
            m_output.syncToHost();
            m_weights.syncToHost();
            m_biases.syncToHost();
            m_beforeActivation.syncToHost();
        }
    }

    __host__ __device__ void forward(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned inOutNi = blockX;
        unsigned inOutCi = blockY;
        unsigned outWi = threadX;
        unsigned outHi = (T_ACTIVATION1 != T_ACTIVATION2) ? threadY * 2 : threadY;

        unsigned iBias = threadY * m_output.w() + outWi;
        unsigned iWeight = m_input.h() * m_input.w() * iBias;
        float fBeforeActivation = m_biases.as<float>(iBias);
        for (unsigned inHi = 0; inHi < m_input.h(); ++inHi)
        {
            for (unsigned inWi = 0; inWi < m_input.w(); ++inWi)
            {
                float fInput = m_input.access<float>(inOutNi, inHi, inWi, inOutCi);
                fBeforeActivation += fInput * m_weights.as<float>(iWeight++);
            }
        }
        m_beforeActivation.access<float>(inOutNi, threadY, outWi, inOutCi) = fBeforeActivation;
        float fAfterActivation = TFunction<T_ACTIVATION1>(fBeforeActivation);
        m_output.access<float>(inOutNi, outHi, outWi, inOutCi) = fAfterActivation;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            float fAfterActivation2 = TFunction<T_ACTIVATION2>(fBeforeActivation);
            m_output.access<float>(inOutNi, outHi + 1, outWi, inOutCi) = fAfterActivation2;
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
TensorRef FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::forward(NvU32 uBatch, TensorRef pInput)
{
    LayerBatchData& batchData = m_batchesData.accessBatchData(uBatch);
    batchData.m_pPrevInput = pInput;
    Tensor<float>& input = *pInput;
    NvU32 n = input.n();
    nvAssert(m_inputDims[0] == 1 && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor<float>& output = *batchData.m_pOutput;
    nvAssert(output.n() == n && m_outputDims[0] == 1 && output.h() == m_outputDims[1] && output.w() == m_outputDims[2] && output.c() == m_outputDims[3]);
    Tensor<float>& beforeActivation = *batchData.m_beforeActivation;

    dim3 grid(n, m_outputDims[3], 1);
    dim3 block(m_outputDims[2], T_ACTIVATION1 == T_ACTIVATION2 ? m_outputDims[1] : m_outputDims[1] / 2, 1);
    FCL_Forward<T_ACTIVATION1, T_ACTIVATION2> forward(input, output, m_weights, m_biases, beforeActivation);
    if (g_bExecuteOnTheGPU)
    {
        fclForwardKernel << <grid, block >> > (forward);
    }
    else
    {
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
    }
    return batchData.m_pOutput;
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FCL_Backward
{
    FCL_Backward(float fBiasesLR, float fWeightsLR, Tensor<float>& input, Tensor<float>& weights, Tensor<float>& biases,
    Tensor<float> &prevLoss, Tensor<float> &loss, Tensor<float> &beforeActivation) :
        m_fBiasesLR(fBiasesLR), m_fWeightsLR(fWeightsLR),
        m_input(input), m_weights(weights), m_biases(biases),
        m_prevLoss(prevLoss), m_loss(loss), m_beforeActivation(beforeActivation)
    {
        if (g_bExecuteOnTheGPU)
        {
            m_input.notifyDeviceBind(false);
            m_weights.notifyDeviceBind(true);
            m_biases.notifyDeviceBind(true);
            m_prevLoss.notifyDeviceBind(true);
            m_loss.notifyDeviceBind(false);
            m_beforeActivation.notifyDeviceBind(false);
        }
        else
        {
            m_input.syncToHost();
            m_weights.syncToHost();
            m_biases.syncToHost();
            m_prevLoss.syncToHost();
            m_loss.syncToHost();
            m_beforeActivation.syncToHost();
        }
    }

    __host__ __device__ void backward(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned outWi = threadX;
        unsigned _outHi = threadY;
        unsigned inWi = blockX;
        unsigned inHi = blockY;

        float fDeltaBias = 0, fDeltaWeight = 0;
        unsigned iWeight = (outWi + _outHi * m_loss.w()) * m_input.h() * m_input.w() + inHi * m_input.w() + inWi;
        float fW = m_weights.as<float>(iWeight);
        for (unsigned inOutNi = 0; inOutNi < m_loss.n(); ++inOutNi)
        {
            for (unsigned inOutCi = 0; inOutCi < m_loss.c(); ++inOutCi)
            {
                unsigned outHi = _outHi * (T_ACTIVATION1 != T_ACTIVATION2 ? 2 : 1);

                float fLoss[2] = {0 , 0};
                fLoss[0] = m_loss.access<float>(inOutNi, outHi, outWi, inOutCi);
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    fLoss[1] = m_loss.access<float>(inOutNi, outHi + 1, outWi, inOutCi);
                }
                if (fLoss[0] == 0 && (T_ACTIVATION1 == T_ACTIVATION2 || fLoss[1] == 0)) // if no error - nothing to do
                    continue;
                float fBeforeActivation = m_beforeActivation.access<float>(inOutNi, _outHi, outWi, inOutCi);
                float fActivationDer = TFunctionDer<T_ACTIVATION1>(fBeforeActivation);
                float fBackwardChain = fLoss[0] * fActivationDer;
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    float fActivation2Der = TFunctionDer<T_ACTIVATION2>(fBeforeActivation);
                    fBackwardChain += fLoss[1] * fActivation2Der;
                }
                fDeltaBias += fBackwardChain;
                // modify the weight corresponding to this summator
                float fInput = m_input.access<float>(inOutNi, inHi, inWi, inOutCi);
                fDeltaWeight += fBackwardChain * fInput;
                if (m_prevLoss.n()) // have we been asked to compute deltaInput?
                {
                    float& fPrevLoss = m_prevLoss.access<float>(inOutNi, inHi, inWi, inOutCi);
                    myAtomicAdd(&fPrevLoss, fBackwardChain * fW);
                }
            }
        }
        // bias address only depends on threadId - meaning the same threadIds from different blocks may race
        unsigned iBias = _outHi * m_loss.w() + outWi;
        // not sure why this division is needed. i added it for the numerical derivative tests to pass
        myAtomicAdd(&m_biases.as<float>(iBias), fDeltaBias* m_fBiasesLR / (m_input.w() * m_input.h()));
        m_weights.as<float>(iWeight) += fDeltaWeight * m_fWeightsLR;
    }

    Tensor<float> m_input, m_weights, m_biases, m_prevLoss, m_loss, m_beforeActivation;
    float m_fBiasesLR = 0, m_fWeightsLR = 0;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
__global__ void fclBackwardKernel(FCL_Backward<T_ACTIVATION1, T_ACTIVATION2> backward)
{
    backward.backward(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
Tensor<float> *FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::backward(NvU32 uBatch, Tensor<float>& loss,
    float fBiasesLR, float fWeightsLR)
{
    auto& batchData = m_batchesData.accessBatchData(uBatch);
    Tensor<float>& input = *batchData.m_pPrevInput;
    NvU32 n = input.n();
    nvAssert(m_inputDims[0] == 1 && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor<float> prevLoss;
    if (batchData.m_pPrevLoss)
    {
        prevLoss = *batchData.m_pPrevLoss;
        nvAssert(prevLoss.n() == n && prevLoss.h() == m_inputDims[1] &&
            prevLoss.w() == m_inputDims[2] && prevLoss.c() == m_inputDims[3]);
    }
    nvAssert(loss.n() == n && m_outputDims[0] == 1 && loss.h() == m_outputDims[1] && loss.w() == m_outputDims[2] && loss.c() == m_outputDims[3]);
    if (prevLoss.n())
    {
        prevLoss.clearSubregion(0, (NvU32)prevLoss.size(), EXECUTE_MODE_DEFAULT);
    }
    Tensor<float>& beforeActivation = *batchData.m_beforeActivation;
    FCL_Backward<T_ACTIVATION1, T_ACTIVATION2> backward(fBiasesLR, fWeightsLR,
        input, m_weights, m_biases, prevLoss, loss, beforeActivation);
    nvAssert(T_ACTIVATION1 == T_ACTIVATION2 || loss.h() % 2 == 0);
    unsigned outHiNum = (T_ACTIVATION1 == T_ACTIVATION2 ? loss.h() : loss.h() / 2);
    dim3 grid(input.w(), input.h(), 1);
    dim3 block(loss.w(), outHiNum, 1);
    if (g_bExecuteOnTheGPU)
    {
        fclBackwardKernel << <grid, block >> > (backward);
    }
    else
    {
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
    }
    return batchData.m_pPrevLoss.get();
}

// explicit instantiation
template struct FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
template struct FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;

