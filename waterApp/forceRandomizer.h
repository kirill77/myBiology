#pragma once

#include <vector>
#include "basics/vectors.h"
#include "MonteCarlo/RNGUniform.h"

template <class T> struct PrContext;

struct ForceRandomizer
{
    void init(PrContext<float>& c);
    void randomize(NvU32 uAtom, PrContext<float>& c);

private:
    RNGUniform m_rng;
    std::vector<rtvector<float, 3>> m_randomDirs;
    double m_fCurSum = 0;
    float m_fAvgForce = 0;
};