#pragma once

// MRELU = RELU mirrored around the origin
enum ACTIVATION { ACTIVATION_RELU, ACTIVATION_MRELU, ACTIVATION_IDENTITY };

template <ACTIVATION T_ACTIVATION>
__host__ __device__ float TFunction(float fInput)
{
    switch (T_ACTIVATION)
    {
    case ACTIVATION_RELU:
        return fInput < 0 ? 0.f : fInput;
    case ACTIVATION_MRELU:
        return fInput > 0 ? 0.f : fInput;
    case ACTIVATION_IDENTITY:
        return fInput;
    default:
        nvAssert(false);
        return 0;
    }
}

template <ACTIVATION T_ACTIVATION>
float TFunctionDer(float fInput)
{
    switch (T_ACTIVATION)
    {
    case ACTIVATION_RELU:
        return fInput < 0 ? 0.f : 1.f;
    case ACTIVATION_MRELU:
        return fInput < 0 ? 1.f : 0.f;
    case ACTIVATION_IDENTITY:
        return 1.f;
    default:
        nvAssert(false);
        return 0;
    }
}
