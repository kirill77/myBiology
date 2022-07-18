#include "neural/tensor.h"
#include "neural/network.h"

// when we bind buffer for device access, we have to make sure GPU memory is all up-to-date
template <class T>
void GPUBuffer<T>::notifyDeviceBind(bool isWriteBind)
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
                cudaMalloc(&m_pDevice, m_nHostElems * sizeof(T));
            }
            m_nDeviceElems = m_nHostElems;
        }
        cudaMemcpy(m_pDevice, m_pHost, m_nHostElems * sizeof(T), cudaMemcpyHostToDevice);
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
    cudaMemcpy(m_pHost, m_pDevice, m_nHostElems * sizeof(T), cudaMemcpyDeviceToHost);
    m_hostRev = m_deviceRev;
}

template <class T>
void GPUBuffer<T>::clearSubregion(NvU32 offset, NvU32 nElemsToClear)
{
    m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
    nvAssert(offset + nElemsToClear <= size());
    memset(&((*m_pOrig)[offset]), 0, nElemsToClear * sizeof(T));
}

template <class T>
template <class SRC_T>
NvU32 GPUBuffer<T>::copySubregionFrom(NvU32 dstOffset, GPUBuffer<SRC_T>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy)
{
    syncToHost();
    src.syncToHost();
    nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
    m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
    NvU32 nDstElems = nSrcElemsToCopy * sizeof(SRC_T) / sizeof(T);
    nvAssert(nDstElems * sizeof(T) == nSrcElemsToCopy * sizeof(SRC_T));
    nvAssert(dstOffset + nDstElems <= size());
    nvAssert(srcOffset + nSrcElemsToCopy <= src.size());
    memcpy(&((*m_pOrig)[dstOffset]), &(src[srcOffset]), nSrcElemsToCopy * sizeof(SRC_T));
    return dstOffset + nDstElems;
}

// explicit instantiations
template NvU32 GPUBuffer<float>::copySubregionFrom(NvU32 dstOffset, GPUBuffer<float>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy);
template struct GPUBuffer<float>;
