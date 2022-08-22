#include "neural/tensor.h"
#include "neural/network.h"
#include "neural/atomsNetwork.h"

// when we bind buffer for device access, we have to make sure GPU memory is all up-to-date
template <class T>
void GPUBuffer<T>::notifyDeviceBind(bool isWriteBind, bool bDiscardPrevContent)
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
                cudaError_t error = cudaMalloc(&m_pDevice, m_nHostElems * sizeof(T));
                nvAssert(error == cudaSuccess);
            }
            m_nDeviceElems = m_nHostElems;
        }
        if (!bDiscardPrevContent)
        {
            cudaMemcpy(m_pDevice, m_pHost, m_nHostElems * sizeof(T), cudaMemcpyHostToDevice);
        }
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
    cudaError_t error = cudaMemcpy(m_pHost, m_pDevice, m_nHostElems * sizeof(T), cudaMemcpyDeviceToHost);
    nvAssert(error == cudaSuccess);
    m_hostRev = m_deviceRev;
}

__global__ void clearKernel(float* p, NvU32 nElemsToClear)
{
    NvU32 uElemToClear = blockIdx.x * blockDim.x + threadIdx.x;
    if (uElemToClear >= nElemsToClear)
        return;
    p[uElemToClear] = 0;
}

static inline bool doesRunOnGPU(EXECUTE_MODE mode)
{
    switch (mode)
    {
    case EXECUTE_MODE_DEFAULT:
        return (RUN_ON_GPU ? true : false);
    case EXECUTE_MODE_FORCE_GPU:
        return true;
    case EXECUTE_MODE_FORCE_CPU:
        return false;
    default:
        nvAssert(false);
        return true;
    }
}

template <class T>
void GPUBuffer<T>::clearSubregion(NvU32 offset, NvU32 nElemsToClear, EXECUTE_MODE mode)
{
    if (doesRunOnGPU(mode))
    {
        m_pOrig->notifyDeviceBind(true, nElemsToClear == m_pOrig->m_nHostElems);
        nvAssert(sizeof(T) == sizeof(float));
        dim3 block(256, 1, 1);
        dim3 grid((nElemsToClear + block.x - 1) / block.x, 1, 1);
        clearKernel << <grid, block >> > (((float*)m_pOrig->m_pDevice) + offset, nElemsToClear);
    }
    else
    {
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        nvAssert(offset + nElemsToClear <= size());
        memset(&((*m_pOrig)[offset]), 0, nElemsToClear * sizeof(T));
    }
}

__global__ void copyKernel(float* pDst, float *pSrc, NvU32 nElems)
{
    NvU32 uElem = blockIdx.x * blockDim.x + threadIdx.x;
    if (uElem >= nElems)
        return;
    pDst[uElem] = pSrc[uElem];
}

template <class T>
template <class SRC_T>
NvU32 GPUBuffer<T>::copySubregionFrom(NvU32 dstOffset, GPUBuffer<SRC_T>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy)
{
    NvU32 nDstElems = nSrcElemsToCopy * sizeof(SRC_T) / sizeof(T);
    nvAssert(nDstElems * sizeof(T) == nSrcElemsToCopy * sizeof(SRC_T));
    nvAssert(dstOffset + nDstElems <= size());
    nvAssert(srcOffset + nSrcElemsToCopy <= src.size());
#if RUN_ON_GPU
    src.notifyDeviceBind(false);
    m_pOrig->notifyDeviceBind(true, src.sizeInBytes() == m_pOrig->sizeInBytes());
    nvAssert(sizeof(T) == sizeof(float) && sizeof(SRC_T) == sizeof(float));
    float* pSrc = ((float*)src.getDevicePointer()) + srcOffset;
    float* pDst = ((float*)getDevicePointer()) + dstOffset;
    dim3 block(256, 1, 1);
    dim3 grid((nDstElems + block.x - 1) / block.x, 1, 1);
    copyKernel<<<grid, block>>>(pDst, pSrc, nDstElems);
#else
    syncToHost();
    src.syncToHost();
    nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
    m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
    memcpy(&((*m_pOrig)[dstOffset]), &(src[srcOffset]), nSrcElemsToCopy * sizeof(SRC_T));
#endif
    return dstOffset + nDstElems;
}

// explicit instantiations
template NvU32 GPUBuffer<float>::copySubregionFrom(NvU32 dstOffset, GPUBuffer<float>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy);
template struct GPUBuffer<float>;
template struct GPUBuffer<ConstantAtomData>;
template struct GPUBuffer<TransientAtomData>;
template struct GPUBuffer<ForceIndices<64>>;
template struct GPUBuffer<ForceValues<64>>;

