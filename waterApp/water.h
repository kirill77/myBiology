#pragma once

#include <algorithm>
#include "basics/bonds.h"
#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGSobol.h"
#include "MonteCarlo/distributions.h"

template <class _T>
struct Water
{
    typedef _T T;

    struct Force
    {
        Force() { }
        Force(NvU32 uAtom1, NvU32 uAtom2) : m_uAtom1(uAtom1), m_uAtom2(uAtom2)
        {
            nvAssert(m_uAtom1 == uAtom1 && m_uAtom2 == uAtom2 && m_uAtom1 != m_uAtom2);
        }
        NvU32 getAtom1Index() const { return m_uAtom1; }
        NvU32 getAtom2Index() const { return m_uAtom2; }

        inline bool operator <(const Force& other) const
        {
            return m_fPotential[1] - m_fPotential[0] < other.m_fPotential[1] - other.m_fPotential[0];
        }

        MyUnits<T> m_fPotential[2];

    private:
        NvU32 m_uAtom1 : 16;
        NvU32 m_uAtom2 : 16;
    };

    struct NODE_DATA // data that we store in each node
    {
    };

    Water()
    {
        m_fBoxSize = MyUnits<T>::angstrom() * 20;
        m_fHalfBoxSize = m_fBoxSize / 2.;
        m_bBox.m_vMin = makeVector<MyUnits<T>, 3>(-m_fHalfBoxSize);
        m_bBox.m_vMax = makeVector<MyUnits<T>, 3>( m_fHalfBoxSize);

        MyUnits<T> volume = m_fBoxSize * m_fBoxSize * m_fBoxSize;
        // one mole of water has volume of 18 milliliters
        NvU32 nWaterMolecules = (NvU32)(AVOGADRO * volume.m_value / MyUnits<T>::milliLiter().m_value / 18);
#ifdef NDEBUG
        m_points.resize(3 * nWaterMolecules);
#else
        // debug can't simulate all molecules - too slow
        m_points.resize(3 * 64);
#endif

        NvU32 nOs = 0, nHs = 0;

        for (NvU32 u = 0; u < m_points.size(); ++u)
        {
            Atom& atom = m_points[u];
            if (nHs < nOs * 2)
            {
                atom.m_nProtons = NPROTONS_H;
                ++nHs;
            }
            else
            {
                atom.m_nProtons = NPROTONS_O;
                ++nOs;
            }

            for (NvU32 uDim = 0; uDim < 3; ++uDim)
            {
                double f = m_rng.generate01();
                atom.m_vPos[0][uDim] = m_bBox.m_vMin[uDim] * f + m_bBox.m_vMax[uDim] * (1 - f);
            }
            m_rng.nextSeed();

            if (!m_bBox.includes(atom.m_vPos[0])) // atom must be inside the bounding box
            {
                __debugbreak();
            }
        }

        m_fWantedAverageKin = MyUnits<T>::fromCelcius(m_fWantedTempC);
        m_fMaxAllowedKin = m_fWantedAverageKin * 10;
        m_fWantedTotalKin = m_fWantedAverageKin * (double)m_points.size();
    }

    struct Atom
    {
        inline MyUnits<T> getMass() const { return BondsDataBase<T>::getAtom(m_nProtons).m_fMass; }
        NvU32 m_nProtons : 8;
        rtvector<MyUnits<T>,3> m_vPos[2], m_vSpeed[2], m_vForce[2];
    };
    inline std::vector<Atom>& points()
    {
        return m_points;
    }

#if 0
    void sortForces()
    {
        std::sort(m_forces.begin(), m_forces.end());
    }
#endif

    void makeTimeStep()
    {
        createListOfForces();

        for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
        {
            m_points[uAtom].m_vForce[0] = rtvector<MyUnits<T>, 3>();
            m_points[uAtom].m_vForce[1] = rtvector<MyUnits<T>, 3>();
        }
        for (NvU32 forceIndex = 0; forceIndex < m_forces.size(); ++forceIndex)
        {
            auto& force = m_forces[forceIndex];
            force.m_fPotential[0] = MyUnits<T>();
            updateForces<0>(force);
        }

        for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
        {
            advectPosition(uAtom, m_fTimeStep);
        }

        // don't let atoms come closer than the bond length - otherwise enormous repulsive force explodes the simulation
        for ( ; ; )
        {
            int nAdjustments = 0;
            for (NvU32 forceIndex = 0; forceIndex < m_forces.size(); ++forceIndex)
            {
                auto& force = m_forces[forceIndex];
                nAdjustments += adjustForceDistance(force);
            }
            // if nothing has been adjusted - break
            if (nAdjustments == 0)
            {
                break;
            }
        }

        for (NvU32 forceIndex = 0; forceIndex < m_forces.size(); ++forceIndex)
        {
            auto& force = m_forces[forceIndex];
            force.m_fPotential[1] = MyUnits<T>();
            updateForces<1>(force);
        }
        for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
        {
            advectSpeed(uAtom, m_fTimeStep);
        }

        // update kinetic energy
        m_fCurTotalKin.clear();
        for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
        {
            auto& point = m_points[uAtom];
            MyUnits<T> fMass = point.getMass();
            point.m_vPos[0] = point.m_vPos[1];
            point.m_vSpeed[0] = point.m_vSpeed[1];
            MyUnits<T> fKin = clampTheSpeed(point.m_vSpeed[0], fMass);
            m_fCurTotalKin += fKin;
        }
    }

    int adjustForceDistance(Force& force)
    {
        NvU32 uAtom1 = force.getAtom1Index();
        auto& atom1 = m_points[uAtom1];
        NvU32 uAtom2 = force.getAtom2Index();
        auto& atom2 = m_points[uAtom2];
        auto& eBond = BondsDataBase<T>::getEBond(atom1.m_nProtons, atom2.m_nProtons, 1);
        auto vDir = computeDir<1>(atom1, atom2);
        auto fDistSqr = lengthSquared(vDir);
        // is distance between the atoms larger than the bonth length? then we don't have to do anything
        if (fDistSqr > eBond.m_fLengthSqr)
            return 0;

        MyUnits<T> fMass1 = atom1.getMass();
        MyUnits<T> fMass2 = atom2.getMass();
        auto fDist = sqrt(fDistSqr);
        // make slightly larger adjustment than necessary to account for floating point errors
        auto fAdjustment = (eBond.m_fLength - sqrt(fDistSqr)) + MyUnits<T>::angstrom() / 1024;
        // massive atom is adjusted by a smaller amount
        double fWeight = removeUnits(fMass2 / (fMass1 + fMass2));
        atom1.m_vPos[1] = wrapThePos(atom1.m_vPos[1] + vDir * (fAdjustment / fDist * fWeight));
        atom2.m_vPos[1] = wrapThePos(atom2.m_vPos[1] - vDir * (fAdjustment / fDist * (1 - fWeight)));

        return 1;
    }

    NvU32 getNNodes() const { return (NvU32)m_ocTree.size(); }
    OcTreeNode<Water>& accessNode(NvU32 index) { return m_ocTree[index]; }
    bool isOkToBeNotLeaf(const OcTreeNode<Water>& node) const
    {
        return node.getNPoints() > 0; // we would we split node with 0 points in it?
    }
    bool canHaveInteraction(const OcTreeNode<Water>& node) const
    {
        return node.getNPoints() > 0;
    }
    void resizeNodes(NvU32 nNodes) { m_ocTree.resize(nNodes); }

    // returns true if after this call interaction between those two boxes are fully accounted for
    bool addLeafAndNodeInteraction(NvU32 leafIndex, const OcBoxStack<T>& leafStack, NvU32 nodeIndex, const OcBoxStack<T>& nodeStack)
    {
        nvAssert(m_ocTree[leafIndex].getNPoints() && m_ocTree[nodeIndex].getNPoints());
        // check if we can treat srcNode as one point as opposed to looking at its individual sub-boxes or points
        if (leafIndex != nodeIndex)
        {
            const auto& leafBox = setUnits<MyUnits<T>>(leafStack.getCurBox());
            const auto& nodeBox = setUnits<MyUnits<T>>(nodeStack.getCurBox());
            // if boxes are too far - particles can't affect each other - rule that interactions are accounted for
            MyUnits<T> fDistSqr;
            for (NvU32 uDim = 0; uDim < 3; ++uDim)
            {
                // add distance in this dimension to the square sum
                if (leafBox.m_vMin[uDim] > nodeBox.m_vMax[uDim])
                    fDistSqr += sqr(leafBox.m_vMin[uDim] - nodeBox.m_vMax[uDim]);
                else if (nodeBox.m_vMin[uDim] > leafBox.m_vMax[uDim])
                    fDistSqr += sqr(nodeBox.m_vMin[uDim] - leafBox.m_vMax[uDim]);
                else continue;

                // if result got too large - this means boxes are too far - bail out
                if (fDistSqr >= BondsDataBase<T>::s_zeroForceDistSqr)
                {
#if ASSERT_ONLY_CODE
                    m_dbgNContributions += 2 * m_ocTree[leafIndex].getNPoints() * m_ocTree[nodeIndex].getNPoints();
#endif
                    return true;
                }
            }
            // we want to descend until leafs because it's possible some nodes will be cut off early that way
            auto& node = m_ocTree[nodeIndex];
            if (!node.isLeaf()) return false;
        }
        auto& leafNode1 = m_ocTree[leafIndex];
        nvAssert(leafNode1.getNPoints());
        auto& leafNode2 = m_ocTree[nodeIndex];
        for (NvU32 uPoint2 = leafNode2.getFirstPoint(); uPoint2 < leafNode2.getEndPoint(); ++uPoint2)
        {
            auto& point2 = m_points[uPoint2];
            for (NvU32 uPoint1 = (leafIndex == nodeIndex) ? uPoint2 + 1 : leafNode1.getFirstPoint(); uPoint1 < leafNode1.getEndPoint(); ++uPoint1)
            {
#if ASSERT_ONLY_CODE
                m_dbgNContributions += 2;
#endif
                auto& point1 = m_points[uPoint1];
                auto vDir = computeDir<0>(point2, point1);
                auto fLengthSqr = lengthSquared(vDir);
                if (fLengthSqr >= BondsDataBase<T>::s_zeroForceDistSqr) // if atoms are too far away - disregard
                {
                    continue;
                }

                NvU32 forceIndex = (NvU32)m_forces.size();
                m_forces.resize(forceIndex + 1);
                m_forces[forceIndex] = Force(uPoint1, uPoint2);
            }
        }
        return true;
    }

    const BBox3<MyUnits<T>>& getBoundingBox() const { return m_bBox; }

    MyUnits<T> evalTemperature() const
    {
        return MyUnits<T>::evalTemperature(m_fCurTotalKin / (NvU32)m_points.size());
    }
    MyUnits<T> evalPressure() const
    {
        return MyUnits<T>::evalPressure(m_fCurTotalKin, m_bBox.evalVolume(), (NvU32)m_points.size());
    }
    const MyUnits<T> &getCurTimeStep() const { return m_fTimeStep; }

    // returns index of first point for which points[u][uDim] >= fSplit
    NvU32 loosePointsSort(NvU32 uBegin, NvU32 uEnd, T fSplit, NvU32 uDim)
    {
        for (; ; ++uBegin)
        {
            nvAssert(uBegin <= uEnd);
            if (uBegin == uEnd)
                return uEnd;
            T f1 = m_points[uBegin].m_vPos[0][uDim].m_value;
            if (f1 < fSplit)
                continue;
            // search for element with which we can swap
            for (--uEnd; ; --uEnd)
            {
                nvAssert(uBegin <= uEnd);
                if (uBegin == uEnd)
                    return uEnd;
                T f2 = m_points[uEnd].m_vPos[0][uDim].m_value;
                if (f2 < fSplit)
                    break;
            }
            nvSwap(m_points[uBegin], m_points[uEnd]);
        }
    }

private:
    // the forces between atoms change rapidly depending on the distance - so time discretization introduces significant errors into simulation.
    // since we can't make time step infinitely small - we compensate for inaccuracies by artificially changing speeds to have constant average
    // temperature
#if 0
    void changeSpeedsToConserveTemp()
    {
#if ASSERT_ONLY_CODE
        // check that speed isn't too high
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            Atom& point = m_points[uPoint];
            MyUnits<T> fMass = point.getMass();
            MyUnits<T> fCurKin = lengthSquared(point.m_vSpeed) * fMass / 2;
            nvAssert(fCurKin.m_value <= m_fMaxAllowedKin.m_value * 1.00001);
        }
#endif
        // multiply everything by a constant to achieve desired average temperature
        T fMultiplier = sqrt(m_fWantedTotalKin.m_value / m_fCurTotalKin.m_value);
#if ASSERT_ONLY_CODE
        MyUnits<T> dbgKin;
#endif
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            Atom& point = m_points[uPoint];
            point.m_vSpeed *= fMultiplier;
#if ASSERT_ONLY_CODE
            dbgKin += lengthSquared(point.m_vSpeed) * point.getMass() / 2;
#endif
        }
        nvAssert(aboutEqual(dbgKin.m_value / m_points.size(), m_fWantedAverageKin.m_value, 0.01));
        m_fCurTotalKin = m_fWantedTotalKin;
    }
#endif
    inline MyUnits<T> clampTheSpeed(rtvector<MyUnits<T>, 3>& vSpeed, MyUnits<T> fMass)
    {
        MyUnits<T> fKin = lengthSquared(vSpeed) * fMass / 2;
        if (fKin <= m_fMaxAllowedKin)
            return fKin;
        vSpeed *= sqrt(m_fMaxAllowedKin / fKin);
#if ASSERT_ONLY_CODE
        fKin = lengthSquared(vSpeed) * fMass / 2;
        nvAssert(aboutEqual(fKin.m_value, m_fMaxAllowedKin.m_value, 0.01));
#endif
        return m_fMaxAllowedKin;
    }
    void createListOfForces()
    {
        // create root oc-tree node
        m_ocTree.resize(1);
        m_ocTree[0].initLeaf(0, (NvU32)m_points.size());

        // initialize stack
        OcBoxStack<T> curStack(0, removeUnits(m_bBox));
        // split oc-tree recursively to get small number of points per leaf
        splitRecursive(curStack);

#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#endif
        m_forces.resize(0);
        m_ocTree[0].addNode2NodeInteractions(0, removeUnits(m_bBox), *this);
        nvAssert(m_dbgNContributions == m_points.size() * (m_points.size() - 1));
    }

#if 0
    void adjustTimeStep()
    {
        bool bDenyStepIncrease = false;
        for (NvU32 uPoint = 0; ; )
        {
            const auto& atom = m_points[uPoint];
            MyUnits<T> fMass = atom.getMass();
            rtvector<MyUnits<T>, 3> vSpeed = atom.m_vSpeed + atom.m_vForce * (m_fTimeStep / fMass);
            rtvector<MyUnits<T>, 3> vDeltaPos = vSpeed * m_fTimeStep;
            MyUnits<T> fDeltaPosSqr = lengthSquared(vDeltaPos);
            if (fDeltaPosSqr > m_fMaxSpaceStepSqr) // if delta pos is too large - decrease step size
            {
                m_fTimeStep *= 0.5;
                uPoint = 0;
                bDenyStepIncrease = true;
                continue;
            }
            if (fDeltaPosSqr * 4 > m_fMaxSpaceStepSqr)
                bDenyStepIncrease = true;
            if (++uPoint >= m_points.size())
            {
                break;
            }
        }
        if (!bDenyStepIncrease)
        {
            m_fTimeStep *= 2;
        }
    }
#endif

    template <NvU32 index>
    rtvector<MyUnits<T>, 3> computeDir(const Atom &atom1, const Atom &atom2) const
    {
        rtvector<MyUnits<T>, 3> vOutDir = atom1.m_vPos[index] - atom2.m_vPos[index];
        for (NvU32 uDim = 0; uDim < 3; ++uDim) // particles positions must wrap around the boundary of bounding box
        {
            if (vOutDir[uDim] < -m_fHalfBoxSize) vOutDir[uDim] += m_fBoxSize;
            else if (vOutDir[uDim] > m_fHalfBoxSize) vOutDir[uDim] -= m_fBoxSize;
        }
        return vOutDir;
    }

    template <NvU32 index>
    inline bool computeForce(const Atom &atom1, const Atom &atom2, rtvector<MyUnits<T>, 3> &vOutDir, typename BondsDataBase<T>::LJ_Out &out) const
    {
        vOutDir = computeDir<index>(atom1, atom2);
        auto& eBond = BondsDataBase<T>::getEBond(atom1.m_nProtons, atom2.m_nProtons, 1);
        return eBond.lennardJones(vOutDir, out);
    }

    template <NvU32 index>
    void updateForces(Force &force)
    {
        NvU32 uAtom1 = force.getAtom1Index();
        auto& atom1 = m_points[uAtom1];
        MyUnits<T> fMass1 = atom1.getMass();
        NvU32 uAtom2 = force.getAtom2Index();
        auto& atom2 = m_points[uAtom2];
        MyUnits<T> fMass2 = atom2.getMass();

        rtvector<MyUnits<T>, 3> vR;
        typename BondsDataBase<T>::LJ_Out out;
        if (computeForce<index>(atom1, atom2, vR, out))
        {
            // symmetric addition ensures conservation of momentum
            atom1.m_vForce[index] += out.vForce;
            atom2.m_vForce[index] -= out.vForce;
            force.m_fPotential[index] = out.fPotential;
        }
    }
    void advectSpeed(NvU32 uAtom, MyUnits<T> fTimeStep)
    {
        auto& atom = m_points[uAtom];
        MyUnits<T> fMass = atom.getMass();
        atom.m_vSpeed[1] = atom.m_vSpeed[0] + (atom.m_vForce[0] + atom.m_vForce[1]) * (fTimeStep / 2 / fMass);
        clampTheSpeed(atom.m_vSpeed[1], fMass);
    }
    // if the atom exits bounding box, it enters from the other side
    rtvector<MyUnits<T>, 3> wrapThePos(const rtvector<MyUnits<T>, 3> &vOldPos)
    {
        auto vNewPos = vOldPos;
        for (NvU32 uDim = 0; uDim < 3; ++uDim)
        {
            if (vNewPos[uDim] < m_bBox.m_vMin[uDim])
            {
                auto fOvershoot = (m_bBox.m_vMin[uDim] - vNewPos[uDim]);
                int nBoxSizes = 1 + (int)removeUnits(fOvershoot / m_fBoxSize);
                vNewPos[uDim] += m_fBoxSize * nBoxSizes;
                nvAssert(m_bBox.m_vMin[uDim] <= vNewPos[uDim] && vNewPos[uDim] <= m_bBox.m_vMax[uDim]);
                continue;
            }
            if (vNewPos[uDim] > m_bBox.m_vMax[uDim])
            {
                auto fOvershoot = (vNewPos[uDim] - m_bBox.m_vMax[uDim]);
                int nBoxSizes = 1 + (int)removeUnits(fOvershoot / m_fBoxSize);
                vNewPos[uDim] -= m_fBoxSize * nBoxSizes;
                nvAssert(m_bBox.m_vMin[uDim] <= vNewPos[uDim] && vNewPos[uDim] <= m_bBox.m_vMax[uDim]);
            }
        }
        nvAssert(m_bBox.includes(vNewPos)); // atom must be inside the bounding box
        return vNewPos;
    }

    void advectPosition(NvU32 uAtom, MyUnits<T> fTimeStep)
    {
        auto& atom = m_points[uAtom];

        MyUnits<T> fMass = atom.getMass();
        auto vAvgSpeed = atom.m_vSpeed[0] + atom.m_vForce[0] * (fTimeStep / 2 / fMass);
        clampTheSpeed(vAvgSpeed, fMass);
        atom.m_vPos[1] = wrapThePos(atom.m_vPos[0] + vAvgSpeed * fTimeStep);
    }

    void splitRecursive(OcBoxStack<T>& stack);

    MyUnits<T> m_fBoxSize, m_fHalfBoxSize;
    BBox3<MyUnits<T>> m_bBox;
    std::vector<Atom> m_points;
    std::vector<Force> m_forces;
    std::vector<OcTreeNode<Water>> m_ocTree;
    RNGSobol m_rng;

    const double m_fWantedTempC = 37;
    MyUnits<T> m_fWantedAverageKin, m_fWantedTotalKin, m_fMaxAllowedKin;
    MyUnits<T> m_fCurTotalKin; // energy conservation variables
    MyUnits<T> m_fTimeStep = MyUnits<T>::nanoSecond() * 0.0000000005;
    MyUnits<T> m_fMaxSpaceStep = MyUnits<T>::nanoMeter() / 512, m_fMaxSpaceStepSqr = m_fMaxSpaceStep * m_fMaxSpaceStep;

#if ASSERT_ONLY_CODE
    NvU64 m_dbgNContributions = 0;
#endif
};