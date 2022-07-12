#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include "MonteCarlo/RNGUniform.h"

#define RUN_ON_GPU 0

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
        return m_pOrig->m_nHostElems;
    }
    size_t sizeInBytes() const
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
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
        for (int i = 0; i < size(); ++i)
        {
            (*m_pOrig)[i] = (T)(rng.generate01() * (fMax - fMin) + fMin);
        }
    }
    template <class SRC_T>
    NvU32 copySubregionFrom(NvU32 dstOffset, GPUBuffer<SRC_T>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy);
    void clearSubregion(NvU32 offset, NvU32 nElemsToClear);
    GPUBuffer<T>(const GPUBuffer<T>& other)
    {
        m_pOrig = other.m_pOrig;
        ++m_pOrig->m_nRefs;
        m_pDevice = m_pOrig->m_pDevice;
        m_nDeviceElems = m_pOrig->m_nDeviceElems;
    }
    void operator =(const GPUBuffer<T> &other)
    {
        if (m_pOrig == other.m_pOrig) return;
        decRef();
        m_pOrig = other.m_pOrig;
        ++m_pOrig->m_nRefs;
        m_pDevice = m_pOrig->m_pDevice;
        m_nDeviceElems = m_pOrig->m_nDeviceElems;
    }
    virtual ~GPUBuffer<T>()
    {
        m_pOrig->decRef();
    }
    void notifyDeviceBind(bool isWriteBind);
    void syncToHost();

public:
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
