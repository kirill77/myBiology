#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <memory>
#include "gpuBuffer.h"

template <class T>
struct Tensor : public GPUBuffer<T>
{
    Tensor() { }
    Tensor(const std::array<unsigned, 4>& dims)
    {
        init(dims);
    }
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

    virtual void serialize(const char* sName, ISerializer& s) override
    {
        std::string sIndent = std::string("Tensor ") + sName;
        std::shared_ptr<Indent> pIndent = s.pushIndent(sIndent.c_str());
        GPUBuffer<T>::serialize(sName, s);
        s.serializeSimpleType("m_dims", m_dims);
    }

private:
    unsigned m_dims[4] = {0};
};

typedef std::shared_ptr<Tensor<float>> TensorRef;

