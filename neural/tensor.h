#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <memory>
#include "gpuBuffer.h"

struct Tensor : public GPUBuffer
{
    Tensor() { }
    Tensor(const std::array<unsigned, 4>& dims, size_t elemSize)
    {
        init(dims, (NvU32)elemSize);
    }
    Tensor(NvU32 n, NvU32 h, NvU32 w, NvU32 c, size_t elemSize)
    {
        init(n, h, w, c , (NvU32)elemSize);
    }
    void init(NvU32 n, NvU32 h, NvU32 w, NvU32 c, size_t elemSize)
    {
        m_dims[0] = n;
        m_dims[1] = h;
        m_dims[2] = w;
        m_dims[3] = c;
        this->resizeWithoutConstructor(n * h * w * c, (NvU32)elemSize);
    }
    void init(const std::array<unsigned, 4>& dims, size_t elemSize)
    {
        init(dims[0], dims[1], dims[2], dims[3], (NvU32)elemSize);
    } 
    unsigned compute1DIndex(NvU32 in, NvU32 ih, NvU32 iw, NvU32 ic) const
    {
        nvAssert(in < n() && ih < h() && iw < w() && ic < c());
        return ic + c() * (iw + w() * (ih + in * h()));
    }
    template <class T>
    T &access(NvU32 in, NvU32 ih, NvU32 iw, NvU32 ic)
    {
        return this->as<T>(compute1DIndex(in, ih, iw, ic));
    }
    template <class T>
    const T& access(NvU32 in, NvU32 ih, NvU32 iw, NvU32 ic) const
    {
        return this->as<T>(compute1DIndex(in, ih, iw, ic));
    }
    unsigned n() const { return m_dims[0]; }
    unsigned h() const { return m_dims[1]; }
    unsigned w() const { return m_dims[2]; }
    unsigned c() const { return m_dims[3]; }
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
    Tensor(const Tensor& other) = delete;
    void operator = (const Tensor& other) = delete;
    unsigned m_dims[4] = {};
};

typedef std::shared_ptr<Tensor> TensorRef;

template <class T>
struct CUDAROTensor : public CUDAROBuffer<T>
{
    CUDAROTensor() { }
    CUDAROTensor(Tensor& t) : CUDAROBuffer<T>(t)
    {
        m_dims[0] = t.n();
        m_dims[1] = t.h();
        m_dims[2] = t.w();
        m_dims[3] = t.c();
    }
    __device__ __host__ unsigned compute1DIndex(NvU32 ni, NvU32 hi, NvU32 wi, NvU32 ci) const
    {
        nvAssert(ni < n() && hi < h() && wi < w() && ci < c());
        return ci + c() * (wi + w() * (hi + ni * h()));
    }
    __device__ __host__ const T& access(unsigned ni, unsigned hi, unsigned wi, unsigned ci) const
    {
        return (*this)[compute1DIndex(ni, hi, wi, ci)];
    }
    __device__ __host__ unsigned n() const { return m_dims[0]; }
    __device__ __host__ unsigned h() const { return m_dims[1]; }
    __device__ __host__ unsigned w() const { return m_dims[2]; }
    __device__ __host__ unsigned c() const { return m_dims[3]; }
private:
    unsigned m_dims[4] = {};
};

template <class T>
struct CUDARWTensor : public CUDARWBuffer<T>
{
    CUDARWTensor() { }
    CUDARWTensor(Tensor& t, bool bDiscardPrevContent) : CUDARWBuffer<T>(t, bDiscardPrevContent)
    {
        m_dims[0] = t.n();
        m_dims[1] = t.h();
        m_dims[2] = t.w();
        m_dims[3] = t.c();
    }
    __device__ __host__ unsigned compute1DIndex(NvU32 ni, NvU32 hi, NvU32 wi, NvU32 ci) const
    {
        nvAssert(ni < n() && hi < h() && wi < w() && ci < c());
        return ci + c() * (wi + w() * (hi + ni * h()));
    }
    __device__ __host__ T& access(unsigned ni, unsigned hi, unsigned wi, unsigned ci)
    {
        return (*this)[compute1DIndex(ni, hi, wi, ci)];
    }
    __device__ __host__ unsigned n() const { return m_dims[0]; }
    __device__ __host__ unsigned h() const { return m_dims[1]; }
    __device__ __host__ unsigned w() const { return m_dims[2]; }
    __device__ __host__ unsigned c() const { return m_dims[3]; }
private:
    unsigned m_dims[4] = {};
};

