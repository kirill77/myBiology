#pragma once

#include "ocTree.h"

struct Neuron : public SimLevel
{
    struct Point
    {
        enum TYPE { P_ION, S_ION, ION_PUMP };

    private:
    };

private:
    float m_fRadiusNM; // it's a circle for simplicity
    
    std::vector<Point> m_points;
};