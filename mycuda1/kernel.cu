#include "neural/tensor.h"
#include "neural/network.h"

#define RUN_ON_CPU 1

template <ACTIVATION T_ACTIVATION1, ACTIVATION T_ACTIVATION2>
struct FullyConnectedLayerCuda
{
    FullyConnectedLayerCuda(Tensor<float>& input, Tensor<float>& output, Tensor<float>& weights, Tensor<float>& biases) :
        m_input(input), m_output(output), m_weights(weights), m_biases(biases)
    {
    }

    void forward(unsigned blockX, unsigned blockY, unsigned threadX, unsigned threadY)
    {
        unsigned inOutNi = blockX;
        unsigned inOutCi = blockY;
        unsigned outWi = threadX;
        unsigned outHi = threadY;

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
#if RUN_ON_CPU
    FullyConnectedLayerCuda<T_ACTIVATION1, T_ACTIVATION2> cudaLayer(input, output, m_weights, m_biases);
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
    
#endif
}

template struct FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
template struct FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;