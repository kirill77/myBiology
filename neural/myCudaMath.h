#pragma once

#ifdef __CUDA_ARCH__
#define myAtomicAdd(p, fValue) atomicAdd(p, fValue)
#else
static inline float myAtomicAdd(float *p, float fValue)
{
    p[0] += fValue; // my C code is single-threaded - so just add without atomic
    return p[0];
}
#endif