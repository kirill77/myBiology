#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include "MonteCarlo/RNGUniform.h"

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

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
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
#ifdef __CUDA_ARCH__
        nvAssert(u < m_nDeviceElems);
        return m_pDevice[u];
#else
        nvAssert(u < m_nHostElems);
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
    size_t size() const
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        return m_pOrig->m_nHostElems;
    }
    size_t sizeInBytes() const
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        return sizeof(T) * m_pOrig->m_nHostElems;
    }
    void resize(size_t nElems)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        m_pOrig->m_pHost = (T*)realloc(m_pOrig->m_pHost, nElems * sizeof(T));
        // call constructors on all new objects
        NvU32 u = m_pOrig->m_nHostElems;
        m_pOrig->m_nHostElems = (NvU32)nElems;
        for ( ; u < nElems; ++u)
        {
            new (&((*this)[u])) T();
        }
    }
    void clearWithRandomValues(T fMin, T fMax)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        RNGUniform rng;
        for (int i = 0; i < size(); ++i)
        {
            (*m_pOrig)[i] = (T)(rng.generate01() * (fMax - fMin) + fMin);
        }
    }
    template <class SRC_T>
    NvU32 copySubregionFrom(NvU32 dstOffset, const GPUBuffer<SRC_T>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        NvU32 nDstElems = nSrcElemsToCopy * sizeof(SRC_T) / sizeof(T);
        nvAssert(nDstElems * sizeof(T) == nSrcElemsToCopy * sizeof(SRC_T));
        nvAssert(dstOffset + nDstElems <= size());
        nvAssert(srcOffset + nSrcElemsToCopy <= src.size());
        memcpy(&((*m_pOrig)[dstOffset]), &(src[srcOffset]), nSrcElemsToCopy * sizeof(SRC_T));
        return dstOffset + nDstElems;
    }
    void clearSubregion(NvU32 offset, NvU32 nElemsToClear)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        nvAssert(offset + nElemsToClear <= size());
        memset(&((*m_pOrig)[offset]), 0, nElemsToClear * sizeof(T));
    }
    GPUBuffer<T>(const GPUBuffer<T>& other)
    {
        m_pOrig = other.m_pOrig;
        ++m_pOrig->m_nRefs;
        m_pDevice = m_pOrig->m_pDevice;
        m_nDeviceElems = m_pOrig->m_nDeviceElems;
    }
    virtual ~GPUBuffer<T>()
    {
        nvAssert(m_pOrig->m_nRefs > 0);
        --m_pOrig->m_nRefs;
    }
    void notifyDeviceBind(bool isWriteBind);
    void syncToHost();


public:
    NvU32 m_hostRev = 0, m_deviceRev = 0;
    T *m_pHost = nullptr, *m_pDevice = nullptr;
    NvU32 m_nHostElems = 0, m_nDeviceElems = 0;
    int m_nRefs = 0;
    GPUBuffer<T>* m_pOrig = nullptr;
};

template <class T>
struct Tensor : public GPUBuffer<T>
{
    Tensor() { }
    void init(unsigned n, unsigned h, unsigned w, unsigned c)
    {
        m_dims[0] = n;
        m_dims[1] = h;
        m_dims[2] = w;
        m_dims[3] = c;
        this->resize(n * h * w * c);
    }
    void init(const std::array<unsigned, 4>& dims)
    {
        init(dims[0], dims[1], dims[2], dims[3]);
    }
    void init(const Tensor<T>& other)
    {
        init(other.getDims());
        this->copySubregionFromBuffer(0, other, 0, other.size());
    }
    __host__ __device__ unsigned compute1DIndex(unsigned in, unsigned ih, unsigned iw, unsigned ic) const
    {
        nvAssert(in < n() && ih < h() && iw < w() && ic < c());
        return ic + c() * (iw + w() * (ih + in * h()));
    }
    __host__ __device__ T &access(unsigned in, unsigned ih, unsigned iw, unsigned ic)
    {
        return (*this)[compute1DIndex(in, ih, iw, ic)];
    }
    __host__ __device__ const T& access(unsigned in, unsigned ih, unsigned iw, unsigned ic) const
    {
        return (*this)[compute1DIndex(in, ih, iw, ic)];
    }
    __device__ __host__ unsigned n() const { return m_dims[0]; }
    __device__ __host__ unsigned h() const { return m_dims[1]; }
    __device__ __host__ unsigned w() const { return m_dims[2]; }
    __device__ __host__ unsigned c() const { return m_dims[3]; }
    std::array<unsigned, 4> getDims() const { return std::array<unsigned, 4>({ m_dims[0], m_dims[1], m_dims[2], m_dims[3] }); }

private:
    unsigned m_dims[4] = {0};
};
