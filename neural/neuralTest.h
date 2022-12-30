#pragma once

#include "network.h"

struct NeuralTest
{
    static void test();
    static bool isTested() { return m_bTested; }

private:
    static bool testRandomDerivative(
        std::shared_ptr<NeuralNetwork> pNetwork,
        std::shared_ptr<Batch> pBatch, NvU32 nChecks);

    static bool m_bTested;
};