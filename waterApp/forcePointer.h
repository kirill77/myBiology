#pragma once

#include <algorithm>
#include "basics/bonds.h"

struct ForcePointer
{
    ForcePointer() : m_uForce(~0U), m_uDist(0) { }
    template <class T>
    ForcePointer(NvU32 forceIndex, MyUnits<T> fDistance) : m_uForce(forceIndex)
    {
        nvAssert(m_uForce == forceIndex);
        NvU32 uDist = (NvU32)(64 * fDistance.m_value / BondsDataBase<T>::s_zeroForceDist.m_value);
        nvAssert(uDist <= 64);
        m_uDist = std::min(uDist, 31U);
    }
    bool operator <=(const ForcePointer& other) const
    {
        return m_uDist >= other.m_uDist;
    }
    bool operator  <(const ForcePointer& other) const
    {
        return m_uDist > other.m_uDist;
    }
    NvU32 getForceIndex() const { return m_uForce; }
private:
    NvU32 m_uForce : 26;
    NvU32 m_uDist : 6;
};

// book-keeping for limited number forces that affect the atom
template <NvU32 N>
struct ForcePointers
{
    bool addForcePointer(ForcePointer f)
    {
        if (m_nForces < N)
        {
            m_forces[m_nForces++] = f;
            if (m_nForces == N) // next force is going to compete with existing forces for slots - need to find the min
            {
                bringMinToFront();
            }
            return true;
        }
        if (f <= m_forces[0]) // if the new force is weaker than the weakest known force
        {
            if (m_nForces >= N) return false; // too many forces already and this one seems not that important
            m_forces[m_nForces++] = m_forces[0];
            m_forces[0] = f;
            return true;
        }
        if (m_nForces >= N)
        {
            m_forces[0] = f; // not enough slots - have to forget the weakest force
            bringMinToFront();
            return true;
        }
        m_forces[m_nForces++] = f;
        return true;
    }
    void clear()
    {
        m_nForces = 0;
    }
    NvU32 size() const { return m_nForces; }
    const ForcePointer &operator[](const NvU32 u) const
    {
        nvAssert(u < m_nForces);
        return m_forces[u];
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
    NvU32 m_nForces = 0;
    ForcePointer m_forces[N];
};
