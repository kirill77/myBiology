#pragma once

struct NeuralTest
{
    static void test();
    static bool isTested() { return m_bTested; }

private:
    static bool m_bTested;
};