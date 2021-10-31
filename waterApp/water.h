#pragma once

#include "forcePointer.h"
#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGSobol.h"
#include "MonteCarlo/distributions.h"

template <class _T>
struct Water
{
    typedef _T T;

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
                atom.m_vPos[uDim] = m_bBox.m_vMin[uDim] * f + m_bBox.m_vMax[uDim] * (1 - f);
            }
            m_rng.nextSeed();

            if (!m_bBox.includes(atom.m_vPos)) // atom must be inside the bounding box
            {
                __debugbreak();
            }
        }

        updateForces();

        m_fInitialPot = m_fCurPot;

        m_fWantedAverageKin = MyUnits<T>::fromCelcius(m_fWantedTempC);
        m_fMaxAllowedKin = m_fWantedAverageKin * 10;
        m_fWantedTotalKin = m_fWantedAverageKin * (double)m_points.size();
    }

    struct Atom
    {
        NvU32 m_nProtons : 8;
        rtvector<MyUnits<T>,3> m_vPos, m_vSpeed, m_vForce;
        ForcePointers<16> m_forcePointers;
    };
    inline std::vector<Atom>& points()
    {
        return m_points;
    }

    void makeTimeStep()
    {
        advect<0>(); // advect velocities by half and positions by full step

        updateForces();

        advect<1>(); // advect velocities by half step

        changeSpeedsToConserveTemp();
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
                auto vDir = point2.m_vPos - point1.m_vPos;
                auto fLengthSqr = lengthSquared(vDir);
                if (fLengthSqr >= BondsDataBase<T>::s_zeroForceDistSqr) // if atoms are too far away - disregard
                {
                    continue;
                }

                NvU32 forceIndex = (NvU32)m_forces.size();
                ForcePointer fp(forceIndex, sqrt(fLengthSqr));
                bool bForceAdded = point1.m_forcePointers.addForcePointer(fp);
                bForceAdded |= point2.m_forcePointers.addForcePointer(fp);
                if (!bForceAdded)
                {
                    // if that force is too weak compared to other forces that those atoms have encountered so far - disregard
                    continue;
                }

                m_forces.resize(forceIndex + 1);
                m_forces[forceIndex] = Force(uPoint1, uPoint2);
            }
        }
        return true;
    }

    const BBox3<MyUnits<T>>& getBoundingBox() const { return m_bBox; }

    MyUnits<T> evalTemperature() const
    {
        return MyUnits<T>::evalTemperature(m_fCurKin / (NvU32)m_points.size());
    }
    MyUnits<T> evalPressure() const
    {
        return MyUnits<T>::evalPressure(m_fCurKin, m_bBox.evalVolume(), (NvU32)m_points.size());
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
            T f1 = m_points[uBegin].m_vPos[uDim].m_value;
            if (f1 < fSplit)
                continue;
            // search for element with which we can swap
            for (--uEnd; ; --uEnd)
            {
                nvAssert(uBegin <= uEnd);
                if (uBegin == uEnd)
                    return uEnd;
                T f2 = m_points[uEnd].m_vPos[uDim].m_value;
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
    void changeSpeedsToConserveTemp()
    {
#if ASSERT_ONLY_CODE
        // check that speed isn't too high
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            Atom& point = m_points[uPoint];
            MyUnits<T> fMass = BondsDataBase<T>::getAtom(point.m_nProtons).m_fMass;
            MyUnits<T> fCurKin = lengthSquared(point.m_vSpeed) * fMass / 2;
            nvAssert(fCurKin.m_value <= m_fMaxAllowedKin.m_value * 1.00001);
        }
#endif
        // multiply everything by a constant to achieve desired average temperature
        T fMultiplier = sqrt(m_fWantedTotalKin.m_value / m_fCurKin.m_value);
#if ASSERT_ONLY_CODE
        MyUnits<T> dbgKin;
#endif
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            Atom& point = m_points[uPoint];
            point.m_vSpeed *= fMultiplier;
#if ASSERT_ONLY_CODE
            dbgKin += lengthSquared(point.m_vSpeed) * BondsDataBase<T>::getAtom(point.m_nProtons).m_fMass / 2;
#endif
        }
        nvAssert(aboutEqual(dbgKin.m_value / m_points.size(), m_fWantedAverageKin.m_value, 0.01));
        m_fCurKin = m_fWantedTotalKin;
    }
    inline MyUnits<T> clampTheSpeed(rtvector<MyUnits<T>, 3> &vSpeed, MyUnits<T> fMass)
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
    // the forces between atoms change rapidly depending on the distance - so time discretization introduces significant errors into simulation. since we can't
    // make time step infinitely small - we compensate for inaccuracies by artificially changing speeds in such a way that total energy of the system is conserved
    void changeSpeedsToConserveEnery()
    {
        MyUnits<T> fEnergyToCompensate = (m_fCurPot + m_fCurKin) - m_fInitialPot;
        // if we have less energy we're supposed to - do nothing
        if (fEnergyToCompensate <= 0)
        {
            return;
        }
        // we have to reduce the speed of some atoms to keep total energy constant
        MyUnits<T> fTotalWeight;
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            fTotalWeight += length(m_points[uPoint].m_vForce);
        }
        m_fCurKin.clear();
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            auto& point = m_points[uPoint];
            MyUnits<T> fCurWeight = length(point.m_vForce) / fTotalWeight;
            MyUnits<T> fCurCompensation = fEnergyToCompensate * fCurWeight;
            MyUnits<T> fCurKineticEnergy = lengthSquared(point.m_vSpeed) * BondsDataBase<T>::getAtom(point.m_nProtons).m_fMass / 2;
            if (fCurKineticEnergy <= fCurCompensation)
            {
                point.m_vSpeed.set(MyUnits<T>(0));
                continue;
            }
            auto fMultiplierSqr = (fCurKineticEnergy - fCurCompensation) / fCurKineticEnergy;
            auto fMultiplier = sqrt(fMultiplierSqr);
            point.m_vSpeed *= fMultiplier;
            m_fCurKin += fCurKineticEnergy * fMultiplierSqr;
        }
    }
    void updateForces()
    {
        // create root oc-tree node
        m_ocTree.resize(1);
        m_ocTree[0].initLeaf(0, (NvU32)m_points.size());

        // initialize stack
        OcBoxStack<T> curStack(0, removeUnits(m_bBox));
        // split oc-tree recursively to get small number of points per leaf
        splitRecursive(curStack);

        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            auto& point = m_points[uPoint];
            point.m_vForce.set(MyUnits<T>(0));
            point.m_forcePointers.clear();
        }

#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#endif
        m_fCurPot.clear(); // prepare for potential energy computation
        m_forces.resize(0);
        m_ocTree[0].addNode2NodeInteractions(0, removeUnits(m_bBox), *this);
        nvAssert(m_dbgNContributions == m_points.size() * (m_points.size() - 1));

        for (NvU32 uForce = 0; uForce < m_forces.size(); ++uForce)
        {
            const auto& force = m_forces[uForce];
            auto& point1 = m_points[force.getAtom1Index()];
            auto& point2 = m_points[force.getAtom2Index()];
            auto& eBond = BondsDataBase<T>::getEBond(point1.m_nProtons, point2.m_nProtons, 1);
            typename BondsDataBase<T>::LJ_Out out;
            out.vForce = point1.m_vPos - point2.m_vPos;
            for (NvU32 uDim = 0; uDim < 3; ++uDim) // particles positions must wrap around the boundary of bounding box
            {
                if (out.vForce[uDim] < -m_fHalfBoxSize) out.vForce[uDim] += m_fBoxSize;
                else if (out.vForce[uDim] > m_fHalfBoxSize) out.vForce[uDim] -= m_fBoxSize;
            }
            if (eBond.lennardJones(out.vForce, out))
            {
                point1.m_vForce += out.vForce;
                point2.m_vForce -= out.vForce;
                m_fCurPot += out.fPotential;
            }
        }
    }

    template <NvU32 VERLET_STEP_INDEX> 
    void advect()
    {
        MyUnits<T> fHalfTimeStep = m_fTimeStep * 0.5;

        // if it's a first VERLET half-step - then do a dry run first to see if we need to increase or decrease the time step
        if (VERLET_STEP_INDEX == 0)
        {
            bool bDenyStepIncrease = false;
            for (NvU32 uPoint = 0; ; )
            {
                const auto& point = m_points[uPoint];
                MyUnits<T> fMass = BondsDataBase<T>::getAtom(point.m_nProtons).m_fMass;
                rtvector<MyUnits<T>, 3> vAcceleration = newtonLaw(fMass, point.m_vForce);
                rtvector<MyUnits<T>, 3> vNewSpeed = point.m_vSpeed + vAcceleration * fHalfTimeStep;
                clampTheSpeed(vNewSpeed, fMass);
                rtvector<MyUnits<T>, 3> vDeltaPos = vNewSpeed * m_fTimeStep;
                MyUnits<T> fDeltaPosSqr = lengthSquared(vDeltaPos);
                if (fDeltaPosSqr > m_fMaxSpaceStepSqr) // if delta pos is too large - decrease step size
                {
                    m_fTimeStep *= 0.5;
                    fHalfTimeStep *= 0.5;
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
                fHalfTimeStep *= 2;
            }
        }

        if (VERLET_STEP_INDEX == 1)
        {
            m_fCurKin.clear(); // prepare for kinetic energy computation
        }

        // change speed of each point according to force and advect points according to new speed
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            auto& point = m_points[uPoint];
            MyUnits<T> fMass = BondsDataBase<T>::getAtom(point.m_nProtons).m_fMass;
            rtvector<MyUnits<T>, 3> vAcceleration = newtonLaw(fMass, point.m_vForce);
            point.m_vSpeed += vAcceleration * fHalfTimeStep;
            MyUnits<T> fKin = clampTheSpeed(point.m_vSpeed, fMass);
            if (VERLET_STEP_INDEX == 1)
            {
                m_fCurKin += fKin;
            }
            if (VERLET_STEP_INDEX == 0)
            {
                auto vDeltaPos = point.m_vSpeed * m_fTimeStep;
                point.m_vPos += vDeltaPos;

                // if the atom exits bounding box, it enters from the other side
                for (NvU32 uDim = 0; uDim < 3; ++uDim)
                {
                    if (point.m_vPos[uDim] < m_bBox.m_vMin[uDim])
                    {
                        auto fOvershoot = (m_bBox.m_vMin[uDim] - point.m_vPos[uDim]);
                        int nBoxSizes = 1 + (int)removeUnits(fOvershoot / m_fBoxSize);
                        point.m_vPos[uDim] += m_fBoxSize * nBoxSizes;
                        nvAssert(m_bBox.m_vMin[uDim] <= point.m_vPos[uDim] && point.m_vPos[uDim] <= m_bBox.m_vMax[uDim]);
                        continue;
                    }
                    if (point.m_vPos[uDim] > m_bBox.m_vMax[uDim])
                    {
                        auto fOvershoot = (point.m_vPos[uDim] - m_bBox.m_vMax[uDim]);
                        int nBoxSizes = 1 + (int)removeUnits(fOvershoot / m_fBoxSize);
                        point.m_vPos[uDim] -= m_fBoxSize * nBoxSizes;
                        nvAssert(m_bBox.m_vMin[uDim] <= point.m_vPos[uDim] && point.m_vPos[uDim] <= m_bBox.m_vMax[uDim]);
                    }
                }
                nvAssert(m_bBox.includes(point.m_vPos)); // atom must be inside the bounding box
            }
        }
    }

    void splitRecursive(OcBoxStack<T>& stack);

    MyUnits<T> m_fBoxSize, m_fHalfBoxSize;
    BBox3<MyUnits<T>> m_bBox;
    std::vector<Atom> m_points;
    struct Force
    {
        Force() { }
        Force(NvU32 uAtom1, NvU32 uAtom2) : m_uAtom1(uAtom1), m_uAtom2(uAtom2), m_isUsed(0)
        {
            nvAssert(m_uAtom1 == uAtom1 && m_uAtom2 == uAtom2 && m_uAtom1 != m_uAtom2);
        }
        void notifyUsed() { m_isUsed = 1; }
        void clearUsed() { m_isUsed = 0; }
        bool isUsed() const { return m_isUsed == 1; }
        NvU32 getAtom1Index() const { return m_uAtom1; }
        NvU32 getAtom2Index() const { return m_uAtom2; }

    private:
        NvU32 m_uAtom1 : 15;
        NvU32 m_uAtom2 : 15;
        NvU32 m_isUsed : 1;
    };
    std::vector<Force> m_forces;
    std::vector<OcTreeNode<Water>> m_ocTree;
    RNGSobol m_rng;

    const double m_fWantedTempC = 37;
    MyUnits<T> m_fWantedAverageKin, m_fWantedTotalKin, m_fMaxAllowedKin;
    MyUnits<T> m_fCurPot, m_fCurKin, m_fInitialPot; // energy conservation variables
    MyUnits<T> m_fTimeStep = MyUnits<T>::nanoSecond() * 0.000001;
    MyUnits<T> m_fMaxSpaceStep = MyUnits<T>::nanoMeter() / 512, m_fMaxSpaceStepSqr = m_fMaxSpaceStep * m_fMaxSpaceStep;

#if ASSERT_ONLY_CODE
    NvU64 m_dbgNContributions;
#endif
};