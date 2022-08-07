#include "neural/tensor.h"
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
void FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::forward(std::vector<TensorRef>& inputs)
{
    nvAssert(inputs.size() == 1 && m_outputs.size() == 1); // this layer has one input tensor and one output tensor
    Tensor<float>& input = *inputs[0];
    nvAssert(input.n() == m_inputDims[0] && input.h() == m_inputDims[1] && input.w() == m_inputDims[2] && input.c() == m_inputDims[3]);
    Tensor<float>& output = *m_outputs[0];
    nvAssert(output.n() == m_outputDims[0] && output.h() == m_outputDims[1] && output.w() == m_outputDims[2] && output.c() == m_outputDims[3]);

    dim3 grid(m_outputDims[0], m_outputDims[3], 1);
    dim3 block(m_outputDims[2], T_ACTIVATION1 == T_ACTIVATION2 ? m_outputDims[1] : m_outputDims[1] / 2, 1);
    FCL_Forward<T_ACTIVATION1, T_ACTIVATION2> forward(input, output, m_weights, m_biases, m_beforeActivation);
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
    FCL_Backward(float fLearningRate, Tensor<float>& input, Tensor<float>& output, Tensor<float>& weights, Tensor<float>& biases,
    Tensor<float> &deltaInput, Tensor<float> &wantedOutput, Tensor<float> &beforeActivation) :
        m_fLearningRate(fLearningRate), m_input(input), m_output(output), m_weights(weights), m_biases(biases),
        m_deltaInput(deltaInput), m_wantedOutput(wantedOutput), m_beforeActivation(beforeActivation)
    {
#if RUN_ON_GPU
        m_input.notifyDeviceBind(false);
        m_output.notifyDeviceBind(false);
        m_weights.notifyDeviceBind(true);
        m_biases.notifyDeviceBind(true);
        m_deltaInput.notifyDeviceBind(true);
        m_wantedOutput.notifyDeviceBind(false);
        m_beforeActivation.notifyDeviceBind(false);
#endif
    }

    __host__ __device__ void backward(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned outWi = threadX;
        unsigned _outHi = threadY;
        unsigned inWi = blockX;
        unsigned inHi = blockY;

        float fDeltaBias = 0;
        for (unsigned inOutNi = 0; inOutNi < m_wantedOutput.n(); ++inOutNi)
        {
            for (unsigned inOutCi = 0; inOutCi < m_wantedOutput.c(); ++inOutCi)
            {
                unsigned outHi = _outHi * (T_ACTIVATION1 == T_ACTIVATION2 ? 1 : 2);

                float fWantedDeltaOut[2] = {0 , 0};
                fWantedDeltaOut[0] = m_wantedOutput.access(inOutNi, outHi, outWi, inOutCi);
                if (m_output.n())
                {
                    fWantedDeltaOut[0] -= m_output.access(inOutNi, outHi, outWi, inOutCi);
                }
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    fWantedDeltaOut[1] = m_wantedOutput.access(inOutNi, outHi + 1, outWi, inOutCi);
                    if (m_output.n())
                    {
                        fWantedDeltaOut[1] -= m_output.access(inOutNi, outHi + 1, outWi, inOutCi);
                    }
                }
                if (fWantedDeltaOut[0] == 0 && (T_ACTIVATION1 == T_ACTIVATION2 || fWantedDeltaOut[1] == 0)) // if no error - nothing to do
                    continue;
                float fBeforeActivation = m_beforeActivation.access(inOutNi, _outHi, outWi, inOutCi);
                float fActivationDer = TFunctionDer<T_ACTIVATION1>(fBeforeActivation);
                float fMult = fWantedDeltaOut[0] * fActivationDer;
                if (T_ACTIVATION1 != T_ACTIVATION2)
                {
                    float fActivation2Der = TFunctionDer<T_ACTIVATION2>(fBeforeActivation);
                    fMult += fWantedDeltaOut[1] * fActivation2Der;
                }
                fMult *= m_fLearningRate;
                fDeltaBias += fMult;
                // modify the weight corresponding to this summator
                unsigned iWeight = (outWi + _outHi * m_wantedOutput.w()) * m_input.h() * m_input.w() + inHi * m_input.w() + inWi;
                float fInput = m_input.access(inOutNi, inHi, inWi, inOutCi);
                float fW = m_weights[iWeight];
                m_weights[iWeight] += fMult * fInput;
                if (m_deltaInput.n()) // have we been asked to compute deltaInput?
                {
                    float& fDeltaInput = m_deltaInput.access(inOutNi, inHi, inWi, inOutCi);
                    fDeltaInput += fMult * (fW + m_weights[iWeight]) / 2;
                }
            }
        }
        // bias address only depends on threadId - meaning the same threadIds from different blocks may race
        unsigned iBias = _outHi * m_wantedOutput.w() + outWi;
        myAtomicAdd(&m_biases[iBias], fDeltaBias);
    }

    Tensor<float> m_input, m_weights, m_biases, m_deltaInput, m_output, m_wantedOutput, m_beforeActivation;
    float m_fLearningRate = 0;
};

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
__global__ void fclBackwardKernel(FCL_Backward<T_ACTIVATION1, T_ACTIVATION2> backward)
{
    backward.backward(blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
void FullyConnectedLayer<T_ACTIVATION1, T_ACTIVATION2>::backward(std::vector<TensorRef>& inputs,
    OUTPUTS_DATA_TYPE outputsDataType, std::vector<TensorRef>& outputsData, float fLearningRate, std::vector<TensorRef>* pDeltaInputs)
{
    nvAssert(inputs.size() == 1);
    Tensor<float>& input = *inputs[0];
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
    Tensor<float> output;
    if (outputsDataType == WANTED_OUTPUTS)
    {
       output = *m_outputs[0];
    }
    nvAssert(wantedOutput.n() == m_outputDims[0] && wantedOutput.h() == m_outputDims[1] && wantedOutput.w() == m_outputDims[2] && wantedOutput.c() == m_outputDims[3]);
    if (deltaInput.n())
    {
        deltaInput.clearSubregion(0, (NvU32)deltaInput.size(), EXECUTE_MODE_DEFAULT);
    }
    FCL_Backward<T_ACTIVATION1, T_ACTIVATION2> backward(fLearningRate, input, output, m_weights, m_biases, deltaInput, wantedOutput, m_beforeActivation);
    nvAssert(T_ACTIVATION1 == T_ACTIVATION2 || wantedOutput.h() % 2 == 0);
    unsigned outHiNum = (T_ACTIVATION1 == T_ACTIVATION2 ? wantedOutput.h() : wantedOutput.h() / 2);
    dim3 grid(input.w(), input.h(), 1);
    dim3 block(wantedOutput.w(), outHiNum, 1);
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

