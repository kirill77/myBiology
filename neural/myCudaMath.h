#pragma once

#ifdef __CUDA_ARCH__
#define myAtomicAdd(p, fValue) atomicAdd(p, fValue)
#else
template <class T>
static inline T myAtomicAdd(T *p, T fValue)
{
    p[0] += fValue; // my C code is single-threaded - so just add without atomic
    return p[0];
}
#endif