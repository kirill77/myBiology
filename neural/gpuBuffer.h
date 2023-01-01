#pragma once

#include <basics/mybasics.h>
#include <basics/serializer.h>
#include <MonteCarlo/RNGUniform.h>

extern bool g_bExecuteOnTheGPU;

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
    }
    template <class T>
    const T& as(NvU32 u) const
    {
        nvAssert(u < m_nHostElems);
        nvAssert(m_hostRev >= m_deviceRev);
        nvAssert(m_elemSize == sizeof(T));
        return ((T*)m_pHost)[u];
    }
    template <class T>
    T& as(NvU32 u)
    {
        nvAssert(m_hostRev >= m_deviceRev);
        nvAssert(m_elemSize == sizeof(T));
        m_hostRev = m_deviceRev + 1;
        nvAssert(u < m_nHostElems);
        return ((T*)m_pHost)[u];
    }
    NvU32 size() const
    {
        return m_nHostElems;
    }
    size_t sizeInBytes() const
    {
        return m_elemSize * m_nHostElems;
    }
    // having this function as a template allows calling T constructor
    template <class T>
    void resize(size_t nElems)
    {
        nvAssert((m_nHostElems * m_elemSize) % sizeof(T) == 0);
        m_nHostElems = (m_nHostElems * m_elemSize) / sizeof(T);
        m_elemSize = sizeof(T);
        resizeInternal<T>(nElems);
    }
    void resizeWithoutConstructor(size_t nElems, NvU32 elemSize)
    {
        m_nHostElems *= m_elemSize;
        m_elemSize = 1;
        resizeInternal<char>(nElems * elemSize);
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
        nvAssert(m_hostRev >= m_deviceRev);
        m_hostRev = m_deviceRev + 1;
        nvAssert(m_elemSize == sizeof(T));
        for (NvU32 i = 0; i < size(); ++i)
        {
            as<T>(i) = (T)(rng.generate01() * (fMax - fMin) + fMin);
        }
    }

    void copySubregionFrom(NvU32 dstOffset, GPUBuffer& src, NvU32 srcOffset, NvU32 nElemsToCopy);

    void clearSubregion(NvU32 offset, NvU32 nElemsToClear, EXECUTE_MODE mode);

    virtual ~GPUBuffer();
    void notifyDeviceBind(bool isWriteBind, bool bDiscardPrevContent = false);
    void syncToHost();

    template <class T>
    T* getDevicePointer() const
    {
        nvAssert(m_elemSize == sizeof(T));
        nvAssert(m_deviceRev >= m_hostRev);
        return (T*)m_pDevice;
    }
    template <class T>
    T* getHostPointer() const
    {
        nvAssert(m_elemSize == sizeof(T));
        nvAssert(m_hostRev >= m_deviceRev);
        return (T*)m_pHost;
    }
    virtual void serialize(const char* sName, ISerializer& s)
    {
        std::string sIndent = std::string("GPUBuffer ") + sName;
        std::shared_ptr<Indent> pIndent = s.pushIndent(sIndent.c_str());
        syncToHost();
        NvU32 nElems = m_nHostElems, elemSize = m_elemSize;
        s.serializeSimpleType("m_nHostElems", nElems);
        s.serializeSimpleType("m_elemSize", elemSize);
        resizeWithoutConstructor(nElems, elemSize);
        s.serializePreallocatedMem("m_pHost", m_pHost, m_elemSize * m_nHostElems);
    }
    double autoReadElem(NvU32 uElem);
    void autoWriteElem(NvU32 uElem, double value);
    NvU32 elemSize() const { return m_elemSize; }

private:
    GPUBuffer(const GPUBuffer& other) = delete;
    void operator = (const GPUBuffer& other) = delete;
    // having this as template allows calling constructor on T - which is what we want
    template <class T>
    void resizeInternal(size_t nElemsNew)
    {
        nvAssert(m_elemSize == sizeof(T));
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
    NvU32 m_hostRev = 0, m_deviceRev = 0;
    void* m_pHost = nullptr, *m_pDevice = nullptr;
    NvU32 m_nHostElems = 0, m_nDeviceElems = 0, m_elemSize = 0;
};

template <class T>
struct CUDAROBuffer
{
    CUDAROBuffer() { }
    CUDAROBuffer(GPUBuffer& b)
    {
        nvAssert(sizeof(T) == b.elemSize());
        if (g_bExecuteOnTheGPU)
        {
            b.notifyDeviceBind(false);
            m_pDevice = b.getDevicePointer<T>();
        }
        else
        {
            b.syncToHost();
            m_pHost = b.getHostPointer<T>();
        }
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
        nvAssert(sizeof(T) == b.elemSize());
        if (g_bExecuteOnTheGPU)
        {
            b.notifyDeviceBind(true, bDiscardPrevContent);
            this->m_pDevice = b.getDevicePointer<T>();
        }
        else
        {
            b.syncToHost();
            this->m_pHost = b.getHostPointer<T>();
        }
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

