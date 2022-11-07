#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <memory>
#include "gpuBuffer.h"

template <class T>
struct Tensor : public GPUBuffer
{
    Tensor() { }
    Tensor(const std::array<unsigned, 4>& dims)
    {
        init(dims);
    }
    void operator =(const Tensor<T>& other)
    {
        GPUBuffer::operator= (other);
        m_dims[0] = other.m_dims[0];
        m_dims[1] = other.m_dims[1];
        m_dims[2] = other.m_dims[2];
        m_dims[3] = other.m_dims[3];
    }
    void init(unsigned n, unsigned h, unsigned w, unsigned c)
    {
        m_dims[0] = n;
        m_dims[1] = h;
        m_dims[2] = w;
        m_dims[3] = c;
        this->resize<T>(n * h * w * c);
    }
    void init(const std::array<unsigned, 4>& dims)
    {
        init(dims[0], dims[1], dims[2], dims[3]);
    } 
    __host__ __device__ unsigned compute1DIndex(unsigned in, unsigned ih, unsigned iw, unsigned ic) const
    {
        nvAssert(in < n() && ih < h() && iw < w() && ic < c());
        return ic + c() * (iw + w() * (ih + in * h()));
    }
    template <class T1>
    __host__ __device__ T1 &access(unsigned in, unsigned ih, unsigned iw, unsigned ic)
    {
        nvAssert(sizeof(T1) == sizeof(T));
        return this->as<T1>(compute1DIndex(in, ih, iw, ic));
    }
    template <class T1>
    __host__ __device__ const T1& access(unsigned in, unsigned ih, unsigned iw, unsigned ic) const
    {
        nvAssert(sizeof(T1) == sizeof(T));
        return this->as<T1>(compute1DIndex(in, ih, iw, ic));
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
    void copyFrom(Tensor<T>& other)
    {
        this->init(other.getDims());
        this->copySubregionFrom(0, other, 0, other.size());
    }

private:
    unsigned m_dims[4] = {0};
};

typedef std::shared_ptr<Tensor<float>> TensorRef;

