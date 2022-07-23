#pragma once

#include "gpuBuffer.h"

enum L2_MODE { L2_MODE_RESET, L2_MODE_ADD };

struct L2Computer
{
    L2Computer() { }

    void accumulateL2Error(GPUBuffer<float>& b1, GPUBuffer<float>& b2, L2_MODE);
    float getAccumulatedError()
    {
        float fError = 0;
        m_pErrors.syncToHost();
        for (NvU32 u = 0; u < m_pErrors.size(); ++u)
            fError += m_pErrors[u];
        nvAssert(fError >= 0 || !isfinite(fError));
        return fError;
    }

private:
    GPUBuffer<float> m_pErrors;
};