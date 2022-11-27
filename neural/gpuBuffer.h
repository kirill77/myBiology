#pragma once

#include <basics/mybasics.h>
#include <basics/serializer.h>
#include <MonteCarlo/RNGUniform.h>

#define RUN_ON_GPU 1

enum EXECUTE_MODE { EXECUTE_MODE_DEFAULT, EXECUTE_MODE_FORCE_GPU, EXECUTE_MODE_FORCE_CPU };

inline void myCheckCudaErrors()
{
#if 1
    extern void _myCheckCudaErrors();
    _myCheckCudaErrors();
#endif
}

struct GPUBuffer
{
    GPUBuffer()
    {
        m_pOrig = this;
        m_pOrig->m_nRefs = 1;
    }
    template <class T>
    __host__ __device__ const T& as(NvU32 u) const
    {
#ifdef __CUDA_ARCH__
        nvAssert(u < m_nDeviceElems);
        return ((T*)m_pDevice)[u];
#else
        nvAssert(u < m_nHostElems);
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        nvAssert(m_pOrig->m_elemSize == sizeof(T));
        return ((T*)m_pOrig->m_pHost)[u];
#endif
    }
    template <class T>
    __host__ __device__ T& as(NvU32 u)
    {
#ifdef __CUDA_ARCH__
        nvAssert(u < m_nDeviceElems);
        return ((T *)m_pDevice)[u];
#else
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        nvAssert(m_pOrig->m_elemSize == sizeof(T));
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        nvAssert(u < m_pOrig->m_nHostElems);
        return ((T*)m_pOrig->m_pHost)[u];
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
        return m_pOrig->m_elemSize * m_pOrig->m_nHostElems;
    }
    // having this function as a template allows calling T constructor
    template <class T>
    void resize(size_t nElems)
    {
        nvAssert((m_nHostElems * m_elemSize) % sizeof(T) == 0);
        m_nHostElems = (m_nHostElems * m_elemSize) / sizeof(T);
        m_elemSize = sizeof(T);
        m_pOrig->resizeInternal<T>(nElems);
    }
    void resizeWithoutConstructor(size_t nElems, NvU32 elemSize)
    {
        m_nHostElems *= m_elemSize;
        m_elemSize = 1;
        m_pOrig->resizeInternal<char>(nElems * elemSize);
        if (m_nHostElems > 0)
        {
            nvAssert(m_nHostElems % elemSize == 0);
            m_nHostElems /= elemSize;
        }
        m_elemSize = elemSize;
    }
    template <class T>
    void clearWithRandomValues(float fMin, float fMax, RNGUniform &rng)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        nvAssert(m_pOrig->m_elemSize == sizeof(T));
        for (NvU32 i = 0; i < size(); ++i)
        {
            m_pOrig->as<T>(i) = (T)(rng.generate01() * (fMax - fMin) + fMin);
        }
    }

    NvU32 copySubregionFrom(NvU32 dstOffset, GPUBuffer& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy);

    void clearSubregion(NvU32 offset, NvU32 nElemsToClear, EXECUTE_MODE mode);

    virtual ~GPUBuffer()
    {
        m_pOrig->decRef();
    }
    void notifyDeviceBind(bool isWriteBind, bool bDiscardPrevContent = false);
    void syncToHost();
    template <class T>
    T* getDevicePointer() const
    {
        nvAssert(m_pOrig->m_elemSize == sizeof(T));
        nvAssert(m_pOrig->m_deviceRev >= m_pOrig->m_hostRev);
        return (T*)m_pDevice;
    }
    virtual void serialize(const char* sName, ISerializer& s)
    {
        std::string sIndent = std::string("GPUBuffer ") + sName;
        std::shared_ptr<Indent> pIndent = s.pushIndent(sIndent.c_str());
        nvAssert(m_pOrig == this);
        syncToHost();
        NvU32 nElems = m_nHostElems, elemSize = m_elemSize;
        s.serializeSimpleType("m_nHostElems", nElems);
        s.serializeSimpleType("m_elemSize", elemSize);
        resizeWithoutConstructor(nElems, elemSize);
        s.serializePreallocatedMem("m_pHost", m_pHost, m_elemSize * m_nHostElems);
    }
    template <class T> T autoReadElem(NvU32 uElem);
    template <class T> void autoWriteElem(NvU32 uElem, T value);
    NvU32 elemSize() const { return m_elemSize; }

private:
    GPUBuffer(const GPUBuffer& other) = delete;
    void operator = (const GPUBuffer& other) = delete;
    void decRef();
    // having this as template allows calling constructor on T - which is what we want
    template <class T>
    void resizeInternal(size_t nElemsNew)
    {
        nvAssert(this == m_pOrig && m_elemSize == sizeof(T));
        m_hostRev = m_deviceRev + 1;
        if (nElemsNew > m_nHostElems)
        {
            T* p = new T[nElemsNew];
            memcpy(p, m_pHost, m_nHostElems * sizeof(T));
            delete[](T*)m_pHost;
            m_pHost = p;
        }
        m_nHostElems = (NvU32)nElemsNew;
    }
    void nullify()
    {
        m_hostRev = 0, m_deviceRev = 0;
        m_pHost = nullptr,  m_pDevice = nullptr;
        m_nHostElems = 0, m_nDeviceElems = 0;
        m_nRefs = 1;
        m_pOrig = this;
    }
    NvU32 m_hostRev = 0, m_deviceRev = 0;
    void* m_pHost = nullptr, *m_pDevice = nullptr;
    NvU32 m_nHostElems = 0, m_nDeviceElems = 0, m_elemSize = 0;
    int m_nRefs = 0;
    GPUBuffer* m_pOrig = nullptr;
};

template <class T>
struct CUDAROBuffer
{
    CUDAROBuffer() { }
    CUDAROBuffer(GPUBuffer& b)
    {
        b.notifyDeviceBind(false);
        m_pDevice = b.getDevicePointer<T>();
        m_size = b.size();
    }
    __device__ __host__ const T& operator[](NvU32 u) const
    {
#ifdef __CUDA_ARCH__
        assert(u < m_size);
        return m_pDevice[u];
#else
        nvAssert(u < m_size);
        return m_pHost[u];
#endif
    }
    __device__ __host__ NvU32 size() const { return m_size; }

protected:
    T* m_pHost = nullptr, *m_pDevice = nullptr;
    NvU32 m_size = 0;
};

template <class T>
struct CUDARWBuffer : public CUDAROBuffer<T>
{
    CUDARWBuffer() { }
    CUDARWBuffer(GPUBuffer &b, bool bDiscardPrevContent)
    {
        b.notifyDeviceBind(true, bDiscardPrevContent);
        this->m_pDevice = b.getDevicePointer<T>();
        this->m_size = b.size();
    }
    __device__ __host__ T& operator[](NvU32 u)
    {
        nvAssert(u < this->m_size);
#ifdef __CUDA_ARCH__
        return m_pDevice[u];
#else
        return this->m_pHost[u];
#endif
    }
};

