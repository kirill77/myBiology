#pragma once

#include "basics/mybasics.h"

// MRELU = RELU mirrored around the origin
enum ACTIVATION { ACTIVATION_RELU, ACTIVATION_MRELU, ACTIVATION_IDENTITY };

template <ACTIVATION T_ACTIVATION, class T>
__host__ __device__ T TFunction(T fInput)
{
    switch (T_ACTIVATION)
    {
    case ACTIVATION_RELU:
        return fInput < 0 ? (T)0 : fInput;
    case ACTIVATION_MRELU:
        return fInput > 0 ? (T)0 : fInput;
    case ACTIVATION_IDENTITY:
        return fInput;
    default:
        nvAssert(false);
        return 0;
    }
}

template <ACTIVATION T_ACTIVATION, class T>
__host__ __device__ T TFunctionDer(T fInput)
{
    switch (T_ACTIVATION)
    {
    case ACTIVATION_RELU:
        return fInput < 0 ? (T)0 : 1.f;
    case ACTIVATION_MRELU:
        return fInput < 0 ? (T)1 : 0.f;
    case ACTIVATION_IDENTITY:
        return 1.f;
    default:
        nvAssert(false);
        return 0;
    }
}
