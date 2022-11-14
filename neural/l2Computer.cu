#include "neural/l2Computer.h"

#if RUN_ON_GPU
// this class collects data from multiple thread of the same block - so it can only run on the GPU
struct CU_LossComputer
{
    static const NvU32 BLOCK_SIZE = 32;

    CU_LossComputer(Tensor& output, Tensor& wantedOutput,
        Tensor& outLoss, GPUBuffer* m_lossPerBlock) : m_output(output),
        m_wantedOutput(wantedOutput), m_outLoss(outLoss)
    {
        if (m_lossPerBlock)
        {
            m_errorStat = *m_lossPerBlock;
            m_errorStat.notifyDeviceBind(true, true);
        }
        nvAssert(wantedOutput.size() == output.size());
        nvAssert(output.size() % output.n() == 0);
        nvAssert(outLoss.size() == output.size());
        m_output.notifyDeviceBind(false);
        m_wantedOutput.notifyDeviceBind(false);
        m_outLoss.notifyDeviceBind(true, true);
    }
    __device__ void computeLoss(int threadX, int blockX, int gridDimX)
    {
        int iStride = gridDimX * BLOCK_SIZE;
        float fSumOfSquares = 0;
        int nElements = 0;
        for (int i = blockX * BLOCK_SIZE + threadX; i < m_output.size(); i += iStride)
        {
            float fDiff = m_wantedOutput.as<float>(i) - m_output.as<float>(i);
            m_outLoss.as<float>(i) = fDiff / (2.f / m_output.size());
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
            m_errorStat.as<float>(blockIdx.x * 2) = fSumOfSquares;
            m_errorStat.as<float>(blockIdx.x * 2 + 1) = nElements;
        }
    }

private:
    Tensor m_output, m_wantedOutput, m_outLoss;
    GPUBuffer m_errorStat;
};
#endif

__global__ void lossKernel(CU_LossComputer lossComputer)
{
    lossComputer.computeLoss(threadIdx.x, blockIdx.x, gridDim.x);
}

void LossComputer::compute(Tensor& output, Tensor& wantedOutput, Tensor& outLoss, float* pErrorStat)
{
    nvAssert(output.getDims() == wantedOutput.getDims());
    nvAssert(output.getDims() == outLoss.getDims());
#if RUN_ON_GPU
    dim3 grid((output.size() + CU_LossComputer::BLOCK_SIZE - 1) / CU_LossComputer::BLOCK_SIZE, 1, 1);
    grid.x = std::min(grid.x, m_lossPerBlock.size() / 2);
    dim3 block(CU_LossComputer::BLOCK_SIZE, 1, 1);
    CU_LossComputer c(output, wantedOutput, outLoss, (pErrorStat == nullptr) ? nullptr : &m_lossPerBlock);
    lossKernel << <grid, block >> > (c);
    if (pErrorStat)
    {
        double fSumOfSquares = 0;
        int nElements = 0;
        m_lossPerBlock.syncToHost();
        for (NvU32 u = 0; u < grid.x; ++u)
        {
            fSumOfSquares += (double)m_lossPerBlock.as<float>(u * 2);
            nElements += (int)m_lossPerBlock.as<float>(u * 2 + 1);
        }
        *pErrorStat = (float)fSumOfSquares;
    }
#else
    m_pErrors.resize(1);
    if (mode == L2_MODE_RESET) m_pErrors[0] = 0;
    b1.syncToHost();
    b2.syncToHost();
    for (NvU32 u = 0; u < b1.size(); ++u)
    {
        m_pErrors[0] += sqr(b1[u] - b2[u]);
    }
#endif
}

