#include "neural/tensor.h"
#include "neural/network.h"
#include "neural/atomsNetwork.h"

__host__ void _myCheckCudaErrors()
{
    cudaError_t status = cudaDeviceSynchronize();
    nvAssert(status == cudaSuccess);
}

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
GPUBuffer::~GPUBuffer()
{
    if (m_pDevice)
    {
        nvAssert(m_elemSize > 0);
        myCudaFree(m_pDevice, m_nDeviceElems * m_elemSize);
    }
    delete[](char*)m_pHost;
    m_pHost = nullptr;
}

void GPUBuffer::syncToHost()
{
    if (m_hostRev >= m_deviceRev)
        return;
    nvAssert(m_nHostElems == m_nDeviceElems);
    cudaError_t error = cudaMemcpy(m_pHost, m_pDevice, m_nHostElems * m_elemSize, cudaMemcpyDeviceToHost);
    nvAssert(error == cudaSuccess);
    m_hostRev = m_deviceRev;
}

template <class T>
__global__ void clearKernel(T* p, NvU32 nElemsToClear)
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
        return g_bExecuteOnTheGPU;
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
    if (doesRunOnGPU(mode))
    {
        notifyDeviceBind(true, nElemsToClear == m_nHostElems);
        dim3 block(256, 1, 1);
        dim3 grid((nElemsToClear + block.x - 1) / block.x, 1, 1);
        if (elemSize() == 4)
        {
            clearKernel << <grid, block >> > (getDevicePointer<float>() + offset, nElemsToClear);
        }
        else
        {
            clearKernel << <grid, block >> > (getDevicePointer<double>() + offset, nElemsToClear);
        }
    }
    else
    {
        nvAssert(offset + nElemsToClear <= size());
        if (elemSize() == 4)
        {
            memset(&(as<float>(offset)), 0, nElemsToClear * elemSize());
        }
        else
        {
            memset(&(as<double>(offset)), 0, nElemsToClear * elemSize());
        }
    }
}

template <class DstType, class SrcType>
__global__ void copyKernel(DstType* pDst, SrcType *pSrc, NvU32 nElems)
{
    NvU32 uElem = blockIdx.x * blockDim.x + threadIdx.x;
    if (uElem >= nElems)
        return;
    pDst[uElem] = (DstType)pSrc[uElem];
}

template <class DstType, class SrcType>
void cpuCopy(DstType* pDst, SrcType* pSrc, NvU32 nElems)
{
    for (NvU32 u = 0; u < nElems; ++u)
    {
        pDst[u] = (DstType)pSrc[u];
    }
}

void GPUBuffer::copySubregionFrom(NvU32 dstOffset, GPUBuffer& src, NvU32 srcOffset, NvU32 nElemsToCopy)
{
    nvAssert(dstOffset + nElemsToCopy <= this->size());
    nvAssert(srcOffset + nElemsToCopy <= src.size());

    if (g_bExecuteOnTheGPU)
    {
        src.notifyDeviceBind(false);
        // (ElemsToCopy == size()) - means we overwrite everything - so can discard the prev content
        notifyDeviceBind(true, nElemsToCopy == size());
        dim3 block(256, 1, 1);
        dim3 grid((nElemsToCopy + block.x - 1) / block.x, 1, 1);
        if (elemSize() == 4)
        {
            float* pDst = getDevicePointer<float>() + dstOffset;
            if (src.elemSize() == 4)
            {
                copyKernel << <grid, block >> > (pDst, src.getDevicePointer<float>() + srcOffset, nElemsToCopy);
            }
            else
            {
                copyKernel << <grid, block >> > (pDst, src.getDevicePointer<double>() + srcOffset, nElemsToCopy);
            }
        }
        else
        {
            double* pDst = getDevicePointer<double>() + dstOffset;
            if (src.elemSize() == 4)
            {
                copyKernel << <grid, block >> > (pDst, src.getDevicePointer<float>() + srcOffset, nElemsToCopy);
            }
            else
            {
                copyKernel << <grid, block >> > (pDst, src.getDevicePointer<double>() + srcOffset, nElemsToCopy);
            }
        }
    }
    else
    {
        syncToHost();
        src.syncToHost();
        nvAssert(m_hostRev >= m_deviceRev);
        m_hostRev = m_deviceRev + 1;
        if (elemSize() == 4)
        {
            float* pDst = &this->as<float>(dstOffset);
            if (src.elemSize() == 4)
            {
                cpuCopy(pDst, &src.as<float>(srcOffset), nElemsToCopy);
            }
            else
            {
                cpuCopy(pDst, &src.as<double>(srcOffset), nElemsToCopy);
            }
        }
        else
        {
            double* pDst = &this->as<double>(dstOffset);
            if (src.elemSize() == 4)
            {
                cpuCopy(pDst, &src.as<float>(srcOffset), nElemsToCopy);
            }
            else
            {
                cpuCopy(pDst, &src.as<double>(srcOffset), nElemsToCopy);
            }
        }
    }
}

double GPUBuffer::autoReadElem(NvU32 uElem)
{
    char buffer[8];
    if (m_deviceRev > m_hostRev)
    {
        cudaError_t result = cudaMemcpy(buffer, (char *)m_pDevice + uElem * elemSize(), elemSize(), cudaMemcpyDeviceToHost);
        nvAssert(result == cudaSuccess);
    }
    else
    {
        memcpy(buffer, (char*)m_pDevice + uElem * elemSize(), elemSize());
    }
    return elemSize() == 4 ? *(float*)buffer : *(double*)buffer;
}

void GPUBuffer::autoWriteElem(NvU32 uElem, double value)
{
    char buffer[8];
    if (elemSize() == 4)
    {
        *(float*)buffer = (float)value;
    }
    else
    {
        *(double*)buffer = value;
    }
    if (m_deviceRev > m_hostRev)
    {
        cudaError_t result = cudaMemcpy((char *)m_pDevice + uElem * elemSize(), buffer, elemSize(), cudaMemcpyHostToDevice);
        nvAssert(result == cudaSuccess);
    }
    else
    {
        memcpy((char*)m_pHost + uElem * elemSize(), buffer, elemSize());
        m_hostRev = m_deviceRev + 1;
    }
}
