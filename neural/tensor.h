#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <memory>
#include "gpuBuffer.h"

struct Tensor : public GPUBuffer
{
    Tensor() { }
    Tensor(const std::array<unsigned, 4>& dims, NvU32 elemSize)
    {
        init(dims, elemSize);
    }
    Tensor(NvU32 n, NvU32 h, NvU32 w, NvU32 c, NvU32 elemSize)
    {
        init(n, h, w, c , elemSize);
    }
    void operator =(const Tensor& other)
    {
        GPUBuffer::operator= (other);
        m_dims[0] = other.m_dims[0];
        m_dims[1] = other.m_dims[1];
        m_dims[2] = other.m_dims[2];
        m_dims[3] = other.m_dims[3];
    }
    void init(NvU32 n, NvU32 h, NvU32 w, NvU32 c, NvU32 elemSize)
    {
        m_dims[0] = n;
        m_dims[1] = h;
        m_dims[2] = w;
        m_dims[3] = c;
        this->resizeWithoutConstructor(n * h * w * c, elemSize);
    }
    void init(const std::array<unsigned, 4>& dims, NvU32 elemSize)
    {
        init(dims[0], dims[1], dims[2], dims[3], elemSize);
    } 
    __host__ __device__ unsigned compute1DIndex(NvU32 in, NvU32 ih, NvU32 iw, NvU32 ic) const
    {
        nvAssert(in < n() && ih < h() && iw < w() && ic < c());
        return ic + c() * (iw + w() * (ih + in * h()));
    }
    template <class T>
    __host__ __device__ T &access(NvU32 in, NvU32 ih, NvU32 iw, NvU32 ic)
    {
        return this->as<T>(compute1DIndex(in, ih, iw, ic));
    }
    template <class T>
    __host__ __device__ const T& access(NvU32 in, NvU32 ih, NvU32 iw, NvU32 ic) const
    {
        return this->as<T>(compute1DIndex(in, ih, iw, ic));
    }
    __device__ __host__ unsigned n() const { return m_dims[0]; }
    __device__ __host__ unsigned h() const { return m_dims[1]; }
    __device__ __host__ unsigned w() const { return m_dims[2]; }
    __device__ __host__ unsigned c() const { return m_dims[3]; }
    std::array<unsigned, 4> getDims() const { return std::array<unsigned, 4>({ m_dims[0], m_dims[1], m_dims[2], m_dims[3] }); }

    virtual void serialize(const char* sName, ISerializer& s) override
    {
        std::string sIndent = std::string("Tensor ") + sName;
        std::shared_ptr<Indent> pIndent = s.pushIndent(sIndent.c_str());
        GPUBuffer::serialize(sName, s);
        s.serializeSimpleType("m_dims", m_dims);
    }
    void copyFrom(Tensor& other)
    {
        this->init(other.getDims(), other.elemSize());
        this->copySubregionFrom(0, other, 0, other.size());
    }

private:
    unsigned m_dims[4] = {};
};

typedef std::shared_ptr<Tensor> TensorRef;

