#pragma once

#include <algorithm>
#include "basics/bonds.h"

struct ForcePointer
{
    ForcePointer() { }
    template <class T>
    ForcePointer(NvU32 forceIndex, MyUnits<T> fDistanceSqr) : m_uForce(forceIndex)
    {
        NvU32 uDist = (NvU32)(32 * sqrt(fDistanceSqr) / BondsDataBase<T>::s_zeroForceDist);
        m_uDist = std::min(uDist, 31);
    }
    bool operator <=(const ForcePointer& other) const
    {
        return m_uDist >= other.m_uDist;
    }
    NvU32 m_uForce : 27;
    NvU32 m_uDist : 5;
};

// book-keeping for limited number forces that affect the atom
template <NvU32 N>
struct AtomForcePointers
{
    void addForcePointer(ForcePointer f)
    {
        if (m_nForces < N)
        {
            m_forces[m_nForces++] = f;
            // next force is going to compete with existing forces for slots - need to find the min
            if (m_nForces == N)
            {
                bringMinToFront();
                return;
            }
        }
        if (f <= m_forces[0]) // if the new force is weaker than weakest known force
        {
            if (m_nForces >= N) return; // too many forces already and this one seems not that important
            m_forces[m_nForces++] = m_forces[0];
            m_forces[0] = f;
            return;
        }
        if (m_nForces >= N)
        {
            m_forces[0] = f; // not enough slots - have to forget the weakest force
            bringMinToFront();
            return;
        }
        m_forces[m_nForces++] = f;
    }
    void clear()
    {
        m_nForces = 0;
    }
    NvU32 size() const { return m_nForces; }
    NvU32 operator[](const NvU32 u) const
    {
        nvAssert(u < m_nForces);
        return m_forces[u].m_forceIndex;
    }
private:
    void bringMinToFront() // put weakest force into slot 0
    {
        ForcePointer tmp = m_forces[0];
        NvU32 uu = 0;
        for (NvU32 u = 1; u < N; ++u)
        {
            if (m_forces[u] < m_forces[0])
            {
                m_forces[0] = m_forces[u];
                uu = u;
            }
        }
        m_forces[uu] = tmp;
    }
    NvU32 m_nForces;
    ForcePointer m_forces[N];
};
