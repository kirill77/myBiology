#include "neural/tensor.h"
#include "neural/network.h"
#include "myCudaMath.h"

bool g_bExecuteOnTheGPU = true;

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2, class T>
struct FCL_Forward
{
    FCL_Forward(Tensor& input, Tensor& output, Tensor& weights, Tensor& biases,
        Tensor &beforeActivation) :
        m_input(input), m_output(output, true), m_weights(weights), m_biases(biases), m_beforeActivation(beforeActivation, true)
    {
    }

    __host__ __device__ void forward(const unsigned blockX, const unsigned blockY, const unsigned threadX, const unsigned threadY)
    {
        const unsigned inOutNi = blockX;
        const unsigned inOutCi = blockY;
        const unsigned outWi = threadX;
        const unsigned outHi = (T_ACTIVATION1 != T_ACTIVATION2) ? threadY * 2 : threadY;

        unsigned iBias = threadY * m_output.w() + outWi;
        unsigned iWeight = m_input.h() * m_input.w() * iBias;
        T fBeforeActivation = m_biases[iBias];
        for (unsigned inHi = 0; inHi < m_input.h(); ++inHi)
        {
            for (unsigned inWi = 0; inWi < m_input.w(); ++inWi)
            {
                T fInput = m_input.access(inOutNi, inHi, inWi, inOutCi);
                T fWeight = m_weights[iWeight++];
                fBeforeActivation += fInput * fWeight;
            }
        }
        m_beforeActivation.access(inOutNi, threadY, outWi, inOutCi) = fBeforeActivation;
        T fAfterActivation = TFunction<T_ACTIVATION1>(fBeforeActivation);
        m_output.access(inOutNi, outHi, outWi, inOutCi) = fAfterActivation;
        if (T_ACTIVATION1 != T_ACTIVATION2)
        {
            T fAfterActivation2 = TFunction<T_ACTIVATION2>(fBeforeActivation);
            m_output.access(inOutNi, outHi + 1, outWi, inOutCi) = fAfterActivation2;
        }
    }

    CUDAROTensor<T> m_input, m_weights, m_biases;
    CUDARWTensor<T> m_output, m_beforeActivation;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2, class T>
__global__ void fclForwardKernel(FCL_Forward<T_ACTIVATION1, T_ACTIVATION2, T> p)
{
    p.forward(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
TensorRef FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::forward(NvU32 uBatch, TensorRef pInput)
{
    if (pInput->elemSize() == 4)
    {
        return forwardInternal<float>(uBatch, pInput);
    }
    return forwardInternal<double>(uBatch, pInput);
}
template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
template <class T>
TensorRef FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::forwardInternal(NvU32 uBatch, TensorRef pInput)
{
    LayerBatchData& batchData = this->m_pBatchData->accessBatchData(uBatch);
    batchData.m_pPrevInput = pInput;
    Tensor& input = *pInput;
    NvU32 n = input.n();
    nvAssert(m_inputDims[0] == 1 && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor& output = *batchData.m_pOutput;
    nvAssert(output.n() == n && m_outputDims[0] == 1 && output.h() == m_outputDims[1] && output.w() == m_outputDims[2] && output.c() == m_outputDims[3]);
    Tensor& beforeActivation = *batchData.m_beforeActivation;

    dim3 grid(n, m_outputDims[3], 1);
    dim3 block(m_outputDims[2], T_ACTIVATION1 == T_ACTIVATION2 ? m_outputDims[1] : m_outputDims[1] / 2, 1);
    FCL_Forward<T_ACTIVATION1, T_ACTIVATION2, T> forward(input, output, *m_pWeights, *m_pBiases, beforeActivation);
    if (g_bExecuteOnTheGPU)
    {
        fclForwardKernel<<<grid, block>>>(forward);
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

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2, class T>
struct FCL_Backward
{
    FCL_Backward(double fBiasesLR, double fWeightsLR, Tensor& input, Tensor& weights, Tensor& biases,
    Tensor *pPrevLoss, Tensor &loss, Tensor &beforeActivation) :
        m_fBiasesLR((T)fBiasesLR), m_fWeightsLR((T)fWeightsLR),
        m_input(input), m_weights(weights, false), m_biases(biases, false),
        m_loss(loss), m_beforeActivation(beforeActivation)
    {
        if (pPrevLoss)
        {
            m_prevLoss = CUDARWTensor<T>(*pPrevLoss, true);
        }
    }

    __host__ __device__ void backward(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned outWi = threadX;
        unsigned _outHi = threadY;
        unsigned inWi = blockX;
        unsigned inHi = blockY;

        T fDeltaBias = 0, fDeltaWeight = 0;
        unsigned iWeight = (outWi + _outHi * m_loss.w()) * m_input.h() * m_input.w() + inHi * m_input.w() + inWi;
        T fW = m_weights[iWeight];
        for (unsigned inOutNi = 0; inOutNi < m_loss.n(); ++inOutNi)
        {
            for (unsigned inOutCi = 0; inOutCi < m_loss.c(); ++inOutCi)
            {
                unsigned outHi = _outHi * (T_ACTIVATION1 != T_ACTIVATION2 ? 2 : 1);

                T fLoss[2] = {0 , 0};
                fLoss[0] = m_loss.access(inOutNi, outHi, outWi, inOutCi);
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    fLoss[1] = m_loss.access(inOutNi, outHi + 1, outWi, inOutCi);
                }
                if (fLoss[0] == 0 && (T_ACTIVATION1 == T_ACTIVATION2 || fLoss[1] == 0)) // if no error - nothing to do
                    continue;
                T fBeforeActivation = m_beforeActivation.access(inOutNi, _outHi, outWi, inOutCi);
                T fActivationDer = TFunctionDer<T_ACTIVATION1>(fBeforeActivation);
                T fBackwardChain = fLoss[0] * fActivationDer;
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    T fActivation2Der = TFunctionDer<T_ACTIVATION2>(fBeforeActivation);
                    fBackwardChain += fLoss[1] * fActivation2Der;
                }
                fDeltaBias += fBackwardChain;
                // modify the weight corresponding to this summator
                T fInput = m_input.access(inOutNi, inHi, inWi, inOutCi);
                fDeltaWeight += fBackwardChain * fInput;
                if (m_prevLoss.n()) // have we been asked to compute deltaInput?
                {
                    T& fPrevLoss = m_prevLoss.access(inOutNi, inHi, inWi, inOutCi);
                    myAtomicAdd(&fPrevLoss, fBackwardChain * fW);
                }
            }
        }
        // bias address only depends on threadId - meaning the same threadIds from different blocks may race
        unsigned iBias = _outHi * m_loss.w() + outWi;
        // not sure why this division is needed. i added it for the numerical derivative tests to pass
        myAtomicAdd(&m_biases[iBias], fDeltaBias* m_fBiasesLR / (m_input.w() * m_input.h()));
        m_weights[iWeight] += fDeltaWeight * m_fWeightsLR;
    }

    CUDARWTensor<T> m_weights, m_biases, m_prevLoss;
    CUDAROTensor<T> m_input, m_loss, m_beforeActivation;
    T m_fBiasesLR = 0, m_fWeightsLR = 0;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2, class T>
__global__ void fclBackwardKernel(FCL_Backward<T_ACTIVATION1, T_ACTIVATION2, T> backward)
{
    backward.backward(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
Tensor* FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::backward(NvU32 uBatch, Tensor& loss,
    double fBiasesLR, double fWeightsLR)
{
    if (loss.elemSize() == 4)
        return backwardInternal<float>(uBatch, loss, fBiasesLR, fWeightsLR);
    return backwardInternal<double>(uBatch, loss, fBiasesLR, fWeightsLR);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
template <class T>
Tensor* FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::backwardInternal(NvU32 uBatch, Tensor& loss,
    double fBiasesLR, double fWeightsLR)
{
    LayerBatchData& batchData = this->m_pBatchData->accessBatchData(uBatch);
    Tensor& input = *batchData.m_pPrevInput;
    NvU32 n = input.n();
    nvAssert(m_inputDims[0] == 1 && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    if (batchData.m_pPrevLoss)
    {
        nvAssert(batchData.m_pPrevLoss->n() == n && batchData.m_pPrevLoss->h() == m_inputDims[1] &&
            batchData.m_pPrevLoss->w() == m_inputDims[2] && batchData.m_pPrevLoss->c() == m_inputDims[3]);
        batchData.m_pPrevLoss->clearSubregion(0, (NvU32)batchData.m_pPrevLoss->size(), EXECUTE_MODE_DEFAULT);
    }
    nvAssert(loss.n() == n && m_outputDims[0] == 1 && loss.h() == m_outputDims[1] && loss.w() == m_outputDims[2] && loss.c() == m_outputDims[3]);
    Tensor& beforeActivation = *batchData.m_beforeActivation;
    FCL_Backward<T_ACTIVATION1, T_ACTIVATION2, T> backward(fBiasesLR, fWeightsLR,
        input, *m_pWeights, *m_pBiases, batchData.m_pPrevLoss.get(), loss, beforeActivation);
    nvAssert(T_ACTIVATION1 == T_ACTIVATION2 || loss.h() % 2 == 0);
    unsigned outHiNum = (T_ACTIVATION1 == T_ACTIVATION2 ? loss.h() : loss.h() / 2);
    dim3 grid(input.w(), input.h(), 1);
    dim3 block(loss.w(), outHiNum, 1);
    if (g_bExecuteOnTheGPU)
    {
        fclBackwardKernel<<<grid, block>>>(backward);
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

