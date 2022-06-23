#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include "MonteCarlo/RNGUniform.h"

template <class T>
struct GPUBuffer
{
    GPUBuffer() { }
    const T& operator[](NvU32 u) const { return m_pHost[u]; }
    T& operator[](NvU32 u) { nvAssert(m_dbgHostRev++ >= m_dbgDeviceRev); return m_pHost[u]; }

    size_t size() const { return m_pHost.size(); }
    size_t sizeInBytes() const { return sizeof(T) * m_pHost.size(); }
    void resize(size_t size)
    {
        m_pHost.resize(size);
        // TODO: resize device buffer as well
    }
    const std::vector<T>& accessHostArray() const { nvAssert(m_dbgHostRev >= m_dbgDeviceRev); return m_pHost; }
    void clearWithRandomValues(T fMin, T fMax)
    {
        RNGUniform rng;
        for (int i = 0; i < size(); ++i)
        {
            (*this)[i] = (T)(rng.generate01() * (fMax - fMin) + fMin);
        }
    }
    template <class SRC_T>
    NvU32 copySubregionFrom(NvU32 dstOffset, const GPUBuffer<SRC_T>& src, NvU32 srcOffset, NvU32 nSrcElemsToCopy)
    {
        NvU32 nDstElems = nSrcElemsToCopy * sizeof(SRC_T) / sizeof(T);
        nvAssert(nDstElems * sizeof(T) == nSrcElemsToCopy * sizeof(SRC_T));
        nvAssert(dstOffset + nDstElems <= size());
        nvAssert(srcOffset + nSrcElemsToCopy <= src.size());
        memcpy(&((*this)[dstOffset]), &(src[srcOffset]), nSrcElemsToCopy * sizeof(SRC_T));
        return dstOffset + nDstElems;
    }
    void clearSubregion(NvU32 offset, NvU32 nElemsToClear)
    {
        nvAssert(offset + nElemsToClear <= size());
        memset(&((*this)[offset]), 0, nElemsToClear * sizeof(T));
    }

private:
    // don't want any copying to happen because it isn't clear what to do with allocated memory
    GPUBuffer<T>(const GPUBuffer<T>& other) { }
#if ASSERT_ONLY_CODE
    NvU32 m_dbgHostRev = 0, m_dbgDeviceRev = 0;
#endif
    std::vector<T> m_pHost;
    NvU32 m_nDeviceElems = 0;
    T* m_pDevice = nullptr;
};

template <class T>
struct Tensor : public GPUBuffer<T>
{
    Tensor() { }
    void init(int n, int h, int w, int c)
    {
        m_dims[0] = n;
        m_dims[1] = h;
        m_dims[2] = w;
        m_dims[3] = c;
        this->resize(n * h * w * c);
    }
    void init(const std::array<int, 4>& dims)
    {
        init(dims[0], dims[1], dims[2], dims[3]);
    }
    void init(const Tensor<T>& other)
    {
        init(other.getDims());
        this->copySubregionFromBuffer(0, other, 0, other.size());
    }
    NvU32 compute1DIndex(int in, int ih, int iw, int ic) const
    {
        return ic + c() * (iw + w() * (ih + in * h()));
    }
    T &access(int in, int ih, int iw, int ic)
    {
        return (*this)[compute1DIndex(in, ih, iw, ic)];
    }
    const T& access(int in, int ih, int iw, int ic) const
    {
        return (*this)[compute1DIndex(in, ih, iw, ic)];
    }
    int n() const { return m_dims[0]; }
    int h() const { return m_dims[1]; }
    int w() const { return m_dims[2]; }
    int c() const { return m_dims[3]; }
    const std::array<int, 4>& getDims() const { return m_dims; }

private:
    std::array<int, 4> m_dims = { -1 };
};