#include "neural/tensor.h"
#include "neural/network.h"
#include "neural/atomsNetwork.h"

size_t g_nCudaBytes = 0;
__host__ cudaError_t myCudaMalloc(void** devPtr, size_t size)
{
    g_nCudaBytes += size;
    auto result = cudaMalloc(devPtr, size);
    nvAssert(result == cudaSuccess);
    return result;
}
__host__ cudaError_t myCudaFree(void* devPtr, size_t size)
{
    nvAssert(g_nCudaBytes >= size);
    g_nCudaBytes -= size;
    auto result = cudaFree(devPtr);
    nvAssert(result == cudaSuccess);
    return result;
}

// when we bind buffer for device access, we have to make sure GPU memory is all up-to-date
void GPUBuffer::notifyDeviceBind(bool isWriteBind, bool bDiscardPrevContent)
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
        nvAssert(m_elemSize > 0);
        if (m_nDeviceElems != m_nHostElems)
        {
            if (m_pDevice)
            {
                myCudaFree(m_pDevice, m_nDeviceElems * m_elemSize);
            }
            if (m_nHostElems == 0)
            {
                m_pDevice = nullptr;
            }
            else
            {
                myCudaMalloc((void **)&m_pDevice, m_nHostElems * m_elemSize);
            }
            m_nDeviceElems = m_nHostElems;
        }
        if (!bDiscardPrevContent)
        {
            cudaMemcpy(m_pDevice, m_pHost, m_nHostElems * m_elemSize, cudaMemcpyHostToDevice);
        }
    }
    m_deviceRev = m_hostRev + (isWriteBind ? 1 : 0);
}
void GPUBuffer::decRef()
{
    nvAssert(this == m_pOrig && m_nRefs > 0);
    if (--m_nRefs == 0)
    {
        if (m_pDevice)
        {
            nvAssert(m_elemSize > 0);
            myCudaFree(m_pDevice, m_nDeviceElems * m_elemSize);
        }
        delete[](char *)m_pHost;
        m_pHost = nullptr;
    }
}

void GPUBuffer::syncToHost()
{
    if (this != m_pOrig)
    {
        m_pOrig->syncToHost();
        return;
    }
    if (m_hostRev >= m_deviceRev)
        return;
    nvAssert(m_nHostElems == m_nDeviceElems);
    cudaError_t error = cudaMemcpy(m_pHost, m_pDevice, m_nHostElems * m_elemSize, cudaMemcpyDeviceToHost);
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

void GPUBuffer::clearSubregion(NvU32 offset, NvU32 nElemsToClear, EXECUTE_MODE mode)
{
    nvAssert(elemSize() == sizeof(float));
    if (doesRunOnGPU(mode))
    {
        m_pOrig->notifyDeviceBind(true, nElemsToClear == m_pOrig->m_nHostElems);
        dim3 block(256, 1, 1);
        dim3 grid((nElemsToClear + block.x - 1) / block.x, 1, 1);
        clearKernel<<<grid, block>>>(((float*)m_pOrig->m_pDevice) + offset, nElemsToClear);
    }
    else
    {
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        nvAssert(offset + nElemsToClear <= size());
        memset(&(m_pOrig->as<float>(offset)), 0, nElemsToClear * elemSize());
    }
}

__global__ void copyKernel(float* pDst, float *pSrc, NvU32 nElems)
{
    NvU32 uElem = blockIdx.x * blockDim.x + threadIdx.x;
    if (uElem >= nElems)
        return;
    pDst[uElem] = pSrc[uElem];
}

NvU32 GPUBuffer::copySubregionFrom(NvU32 dstOffset, GPUBuffer& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy)
{
    NvU32 nDstElems = nSrcElemsToCopy * src.elemSize() / elemSize();
    nvAssert(nDstElems * elemSize() == nSrcElemsToCopy * src.elemSize());
    nvAssert(dstOffset + nDstElems <= size());
    nvAssert(srcOffset + nSrcElemsToCopy <= src.size());
#if RUN_ON_GPU
    src.notifyDeviceBind(false);
    m_pOrig->notifyDeviceBind(true, src.sizeInBytes() == m_pOrig->sizeInBytes());
    nvAssert(elemSize() == sizeof(float) && src.elemSize() == sizeof(float));
    float* pSrc = src.getDevicePointer<float>() + srcOffset;
    float* pDst = getDevicePointer<float>() + dstOffset;
    dim3 block(256, 1, 1);
    dim3 grid((nDstElems + block.x - 1) / block.x, 1, 1);
    copyKernel<<<grid, block>>>(pDst, pSrc, nDstElems);
#else
    syncToHost();
    src.syncToHost();
    nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
    m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
    memcpy(&((*m_pOrig)[dstOffset]), &(src[srcOffset]), nSrcElemsToCopy * src.elemSize());
#endif
    return dstOffset + nDstElems;
}

template <class T>
T GPUBuffer::autoReadElem(NvU32 uElem)
{
    if (m_pOrig != this)
        return m_pOrig->autoReadElem<T>(uElem);
    nvAssert(m_elemSize == sizeof(T));
    if (m_deviceRev > m_hostRev)
    {
        T value;
        cudaError_t result = cudaMemcpy(&value, (T*)m_pDevice + uElem, sizeof(T), cudaMemcpyDeviceToHost);
        nvAssert(result == cudaSuccess);
        return value;
    }
    return ((T*)m_pHost)[uElem];
}

template <class T>
void GPUBuffer::autoWriteElem(NvU32 uElem, T value)
{
    if (m_pOrig != this)
    {
        m_pOrig->autoWriteElem(uElem, value);
        return;
    }
    nvAssert(m_elemSize == sizeof(T));
    if (m_deviceRev > m_hostRev)
    {
        cudaError_t result = cudaMemcpy((T*)m_pDevice + uElem, &value, sizeof(T), cudaMemcpyHostToDevice);
        nvAssert(result == cudaSuccess);
    }
    else
    {
        ((T*)m_pHost)[uElem] = value;
        m_hostRev = m_deviceRev + 1;
    }
}

// explicit instantiations
template float GPUBuffer::autoReadElem(NvU32 uElem);
template double GPUBuffer::autoReadElem(NvU32 uElem);
template void GPUBuffer::autoWriteElem(NvU32 uElem, float value);
template void GPUBuffer::autoWriteElem(NvU32 uElem, double value);
