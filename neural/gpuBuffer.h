#pragma once

#include <basics/mybasics.h>
#include <basics/serializer.h>
#include <MonteCarlo/RNGUniform.h>

#define RUN_ON_GPU 1

enum EXECUTE_MODE { EXECUTE_MODE_DEFAULT, EXECUTE_MODE_FORCE_GPU, EXECUTE_MODE_FORCE_CPU };

template <class T>
struct GPUBuffer
{
    GPUBuffer()
    {
        m_pOrig = this;
        m_pOrig->m_nRefs = 1;
    }
    __host__ __device__ const T& operator[](NvU32 u) const
    {
#ifdef __CUDA_ARCH__
        nvAssert(u < m_nDeviceElems);
        return m_pDevice[u];
#else
        nvAssert(u < m_nHostElems);
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        return m_pOrig->m_pHost[u];
#endif
    }
    __host__ __device__ T& operator[](NvU32 u)
    {
#ifdef __CUDA_ARCH__
        nvAssert(u < m_nDeviceElems);
        return m_pDevice[u];
#else
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        nvAssert(u < m_pOrig->m_nHostElems);
        return m_pOrig->m_pHost[u];
#endif
    }
    __host__ __device__ NvU32 size() const
    {
#ifdef __CUDA_ARCH__
        return m_nHostElems;
#else
        return m_pOrig->m_nHostElems;
#endif
    }
    size_t sizeInBytes() const
    {
        return sizeof(T) * m_pOrig->m_nHostElems;
    }
    void resize(size_t nElems)
    {
        m_pOrig->resizeInternal(nElems);
    }
    void clearWithRandomValues(T fMin, T fMax)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        RNGUniform rng;
        for (NvU32 i = 0; i < size(); ++i)
        {
            (*m_pOrig)[i] = (T)(rng.generate01() * (fMax - fMin) + fMin);
        }
    }
    template <class SRC_T>
    NvU32 copySubregionFrom(NvU32 dstOffset, GPUBuffer<SRC_T>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy);

    void clearSubregion(NvU32 offset, NvU32 nElemsToClear, EXECUTE_MODE mode);

    GPUBuffer<T>(const GPUBuffer<T>& other)
    {
        copyFrom(other);
    }
    void operator =(const GPUBuffer<T>& other)
    {
        if (m_pOrig == other.m_pOrig) return;
        decRef();
        copyFrom(other);
    }
    virtual ~GPUBuffer<T>()
    {
        m_pOrig->decRef();
    }
    void notifyDeviceBind(bool isWriteBind, bool bDiscardPrevContent = false);
    void syncToHost();
    T* getDevicePointer() const
    {
        nvAssert(m_pOrig->m_deviceRev >= m_pOrig->m_hostRev);
        return m_pDevice;
    }
    void serialize(ISerializer& s)
    {
        nvAssert(m_pOrig == this);
        nvAssert(m_hostRev >= m_deviceRev);
        NvU32 nElems = m_nHostElems;
        s.serializePreallocatedMem(&nElems, sizeof(nElems));
        resize(nElems);
        s.serializePreallocatedMem(m_pHost, sizeof(T) * m_nHostElems);
    }

public:
    void copyFrom(const GPUBuffer<T>& other)
    {
        m_pOrig = other.m_pOrig;
        ++m_pOrig->m_nRefs;
        m_pDevice = m_pOrig->m_pDevice;
        m_nHostElems = m_pOrig->m_nHostElems;
        m_nDeviceElems = m_pOrig->m_nDeviceElems;
    }
    void decRef()
    {
        nvAssert(this == m_pOrig && m_nRefs > 0);
        if (--m_nRefs == 0)
        {
            delete[]m_pHost;
            m_pHost = nullptr;
        }
    }
    void resizeInternal(size_t nElemsNew)
    {
        nvAssert(this == m_pOrig);
        if (nElemsNew == m_nHostElems)
            return;
        m_hostRev = m_deviceRev + 1;
        if (nElemsNew > m_nHostElems)
        {
            T* p = new T[nElemsNew];
            memcpy(p, m_pHost, m_nHostElems * sizeof(T));
            delete[]m_pHost;
            m_pHost = p;
        }
        m_nHostElems = (NvU32)nElemsNew;
    }
    NvU32 m_hostRev = 0, m_deviceRev = 0;
    T* m_pHost = nullptr, * m_pDevice = nullptr;
    NvU32 m_nHostElems = 0, m_nDeviceElems = 0;
    int m_nRefs = 0;
    GPUBuffer<T>* m_pOrig = nullptr;
};

