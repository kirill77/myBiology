#include "neural/l2Computer.h"

// this class collects data from multiple thread of the same block - so it can only run on the GPU
template <class T>
struct CU_LossComputer
{
    static const NvU32 BLOCK_SIZE = 32;

    CU_LossComputer(Tensor& output, Tensor& wantedOutput,
        Tensor& outLoss, GPUBuffer* pLossPerBlock) : m_output(output),
        m_wantedOutput(wantedOutput), m_outLoss(outLoss, true)
    {
        if (pLossPerBlock)
        {
            m_errorStat = CUDARWBuffer<T>(*pLossPerBlock, true);
        }
        nvAssert(wantedOutput.size() == output.size());
        nvAssert(output.size() % output.n() == 0);
        nvAssert(outLoss.size() == output.size());
    }
    __device__ void computeLoss(int threadX, int blockX, int gridDimX)
    {
        int iStride = gridDimX * BLOCK_SIZE;
        T fSumOfSquares = 0;
        int nElements = 0;
        for (int i = blockX * BLOCK_SIZE + threadX; i < m_output.size(); i += iStride)
        {
            T fDiff = m_wantedOutput[i] - m_output[i];
            m_outLoss[i] = fDiff / (m_output.size() / 2.f);
            fSumOfSquares += sqr(fDiff) / m_output.size();
            ++nElements;
        }
        if (m_errorStat.size() == 0)
            return;
        // collect errors from all threads of the block
        const NvU32 FULL_MASK = 0xffffffff;
        for (int offset = 16; offset > 0; offset /= 2)
        {
            fSumOfSquares += __shfl_down_sync(FULL_MASK, fSumOfSquares, offset);
            nElements += __shfl_down_sync(FULL_MASK, nElements, offset);
        }
        if (threadX == 0)
        {
            m_errorStat[blockIdx.x * 2] = fSumOfSquares;
            m_errorStat[blockIdx.x * 2 + 1] = nElements;
        }
    }

private:
    CUDARWTensor<T> m_outLoss;
    CUDAROTensor<T> m_output, m_wantedOutput;
    CUDARWBuffer<T> m_errorStat;
};

template <class T>
__global__ void lossKernel(CU_LossComputer<T> lossComputer)
{
    lossComputer.computeLoss(threadIdx.x, blockIdx.x, gridDim.x);
}

void LossComputer::compute(Tensor& output, Tensor& wantedOutput, Tensor& outLoss, double* pErrorStat)
{
    nvAssert(output.getDims() == wantedOutput.getDims());
    nvAssert(output.getDims() == outLoss.getDims());
    if (output.elemSize() == 4)
    {
        computeInternal<float>(output, wantedOutput, outLoss, pErrorStat);
    }
    else
    {
        computeInternal<double>(output, wantedOutput, outLoss, pErrorStat);
    }
}

template <class T>
void LossComputer::computeInternal(Tensor& output, Tensor& wantedOutput, Tensor& outLoss, double* pErrorStat)
{
    dim3 grid((output.size() + CU_LossComputer<T>::BLOCK_SIZE - 1) / CU_LossComputer<T>::BLOCK_SIZE, 1, 1);
    grid.x = std::min(grid.x, m_lossPerBlock.size() / 2);
    dim3 block(CU_LossComputer<T>::BLOCK_SIZE, 1, 1);
    CU_LossComputer<T> c(output, wantedOutput, outLoss, (pErrorStat == nullptr) ? nullptr : &m_lossPerBlock);
    double fSumOfSquares = 0;
    int nElements = 0;
    if (g_bExecuteOnTheGPU)
    {
        lossKernel<<<grid, block>>>(c);
        if (pErrorStat)
        {
            m_lossPerBlock.syncToHost();
            for (NvU32 u = 0; u < grid.x; ++u)
            {
                fSumOfSquares += (double)m_lossPerBlock.as<T>(u * 2);
                nElements += (int)m_lossPerBlock.as<T>(u * 2 + 1);
            }
            *pErrorStat = fSumOfSquares;
        }
    }
    else
    {
        output.syncToHost();
        wantedOutput.syncToHost();
        for (NvU32 i = 0; i < outLoss.size(); ++i)
        {
            double fOutput = output.as<T>(i);
            double fDiff = wantedOutput.as<T>(i) - fOutput;
            outLoss.as<T>(i) = (T)(fDiff / (output.size() / 2.));
            fSumOfSquares += sqr(fDiff) / output.size();
            ++nElements;
        }
        if (pErrorStat)
        {
            *pErrorStat = fSumOfSquares;
        }
    }
}

