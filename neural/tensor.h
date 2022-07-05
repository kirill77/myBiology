#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include "MonteCarlo/RNGUniform.h"

template <class T>
struct GPUBuffer
{
    GPUBuffer()
    {
        m_pOrig = this;
        m_pOrig->m_nRefs = 1;
    }
    const T& operator[](NvU32 u) const
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        return m_pOrig->m_pHost[u];
    }
    T& operator[](NvU32 u)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        return m_pOrig->m_pHost[u];
    }
    size_t size() const
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        return m_pOrig->m_pHost.size();
    }
    size_t sizeInBytes() const
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        return sizeof(T) * m_pOrig->m_pHost.size();
    }
    void resize(size_t size)
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev && m_pOrig->m_nRefs == 1);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        m_pOrig->m_pHost.resize(size);
    }
    const std::vector<T>& accessHostArray() const
    {
        nvAssert(m_pOrig->m_hostRev >= m_pOrig->m_deviceRev);
        m_pOrig->m_hostRev = m_pOrig->m_deviceRev + 1;
        return m_pOrig->m_pHost;
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
        m_nDeviceElems = other.m_nDeviceElems;
        m_pDevice = other.m_pDevice;
        m_pOrig = other.m_pOrig;
        ++m_pOrig->m_nRefs;
    }
    virtual ~GPUBuffer<T>()
    {
        nvAssert(m_pOrig->m_nRefs > 0);
        --m_pOrig->m_nRefs;
    }

private:
    NvU32 m_hostRev = 0, m_deviceRev = 0;
    std::vector<T> m_pHost;
    NvU32 m_nDeviceElems = 0;
    T* m_pDevice = nullptr;
    GPUBuffer<T>* m_pOrig = nullptr;
    int m_nRefs = 0;
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
    unsigned compute1DIndex(unsigned in, unsigned ih, unsigned iw, unsigned ic) const
    {
        nvAssert(in < n() && ih < h() && iw < w() && ic < c());
        return ic + c() * (iw + w() * (ih + in * h()));
    }
    T &access(unsigned in, unsigned ih, unsigned iw, unsigned ic)
    {
        return (*this)[compute1DIndex(in, ih, iw, ic)];
    }
    const T& access(unsigned in, unsigned ih, unsigned iw, unsigned ic) const
    {
        return (*this)[compute1DIndex(in, ih, iw, ic)];
    }
    unsigned n() const { return m_dims[0]; }
    unsigned h() const { return m_dims[1]; }
    unsigned w() const { return m_dims[2]; }
    unsigned c() const { return m_dims[3]; }
    const std::array<unsigned, 4>& getDims() const { return m_dims; }

private:
    std::array<unsigned, 4> m_dims = { 0 };
};
