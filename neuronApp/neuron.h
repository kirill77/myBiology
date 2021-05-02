#pragma once

#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGUniform.h"
#include "MonteCarlo/distributions.h"

template <class T>
struct Neuron
{
    Neuron() : m_rng((NvU32)time(nullptr))
    {
        // create IONS
        for (NvU32 u = 0; u < 1024; ++u)
        {
            Point point;
            if ((u % 100) < 50) // percentage of K ions
            {
                point.m_flags |= FLAG_K_ION;
            }
            else
            {
                point.m_flags |= FLAG_NA_ION;
            }
            for (NvU32 u = 0; u < 3; ++u)
            {
                point.m_pos[u] = m_rng.generate01();
            }
            point.m_pos = SphereVolumeDistribution<T>::generate(point.m_pos);
            m_points.push_back(point);
        }
    }

    enum FLAGS { FLAG_ION_PUMP = 1, FLAG_K_ION = 2, FLAG_NA_ION = 4 };
    struct Point
    {
        NvU32 m_flags = 0;
        rtvector<T, 3> m_pos;
    };
    inline std::vector<Point>& points()
    {
        return m_points;
    }

    void makeTimeStep(double fDeltaT_MS)
    {
        for (NvU32 u = 0; u < m_points.size(); ++u)
        {
            m_points[u].m_pos[0] += (T)((m_rng.generate01() * 2 - 1) * 0.01);
            m_points[u].m_pos[1] += (T)((m_rng.generate01() * 2 - 1) * 0.01);
            m_points[u].m_pos[2] += (T)((m_rng.generate01() * 2 - 1) * 0.01);
        }
    }

private:
    float m_fRadiusMicroMeters = 100; // it's a circle for simplicity
    std::vector<Point> m_points;
    OcTreeNode<Neuron> m_ocTree;
    RNGUniform m_rng;
};