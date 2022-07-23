#include "neural/l2Computer.h"

__global__ void l2ErrorKernel(GPUBuffer<float> p1, GPUBuffer<float> p2, GPUBuffer<float> pDst, L2_MODE mode)
{
    NvU32 uElem = blockIdx.x * blockDim.x + threadIdx.x;
    if (uElem >= p1.size())
        return;
    float fError = sqr(p1[uElem] - p2[uElem]);
    // collect error from all threads
    const NvU32 FULL_MASK = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        fError += __shfl_down_sync(FULL_MASK, fError, offset);
    }

    // 0-th thread writes the error accumulated for the whole block
    if (threadIdx.x == 0)
    {
        if (mode == L2_MODE_RESET)
            pDst[blockIdx.x] = fError;
        else pDst[blockIdx.x] += fError;
    }
}

void L2Computer::accumulateL2Error(GPUBuffer<float>& b1, GPUBuffer<float>& b2, L2_MODE mode)
{
    nvAssert(b1.size() == b2.size());
#if RUN_ON_GPU
    dim3 grid((b1.size() + 31) / 32, 1, 1);
    dim3 block(32, 1, 1);
    b1.notifyDeviceBind(false);
    b2.notifyDeviceBind(false);
    m_pErrors.resize(grid.x);
    m_pErrors.notifyDeviceBind(true, (mode == L2_MODE_RESET));
    l2ErrorKernel << <grid, block >> > (b1, b2, m_pErrors, mode);
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

