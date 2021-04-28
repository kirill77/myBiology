#pragma once

#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGUniform.h"
#include "MonteCarlo/distributions.h"

template <class T>
struct Neuron
{
    Neuron()
    {
        RNGUniform rng((NvU32)time(nullptr));

        // create IONS
        for (NvU32 u = 0; u < 1024; ++u)
        {
            Point point;
            point.m_flags = Point::FLAG_ION;
            if ((u % 100) < 50) // percentage of K ions
            {
                point.m_flags |= Point::FLAG_K;
            }
            else
            {
                point.m_flags |= Point::FLAG_NA;
            }
            for (NvU32 u = 0; u < 3; ++u)
            {
                point.m_pos[u] = rng.generate01();
            }
            point.m_pos = SphereVolumeDistribution<T>::generate(point.m_pos);
            m_points.push_back(point);
        }
    }

    struct Point
    {
        enum FLAGS { FLAG_ION = 1, FLAG_ION_PUMP = 2, FLAG_K, FLAG_NA };
        NvU32 m_flags = 0;
        rtvector<T, 3> m_pos;
    };
    inline std::vector<Point>& points()
    {
        return m_points;
    }

private:
    float m_fRadiusMicroMeters = 100; // it's a circle for simplicity
    std::vector<Point> m_points;
};