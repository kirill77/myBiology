#pragma once

#include "basics/bonds.h"
#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGUniform.h"
#include "MonteCarlo/distributions.h"

template <class _T>
struct Water
{
    typedef _T T;

    struct NODE_DATA // data that we store in each node
    {
        MyUnits<double> m_fTotalCharge;
    };

    Water() : m_rng(1274)//(NvU32)time(nullptr))
    {
        m_fBoxSize = MyUnits<T>::angstrom() * 10;
        m_fHalfBoxSize = m_fBoxSize / 2.;
        m_bBox.m_vMin = makeVector<MyUnits<T>, 3>(-m_fHalfBoxSize);
        m_bBox.m_vMax = makeVector<MyUnits<T>, 3>( m_fHalfBoxSize);

        m_points.resize(64);// 1000 * 3);
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

            // convert u into 3 coordinates (space-filling curve)
            NvU32 uX = 0, uY = 0, uZ = 0;
            for (NvU32 uBit = 0, _u = u; _u > 0; ++uBit)
            {
                uX |= (_u & 1) ? (1 << uBit) : 0;
                uY |= (_u & 2) ? (1 << uBit) : 0;
                uZ |= (_u & 4) ? (1 << uBit) : 0;
                _u >>= 3;
            }
            atom.m_vPos[0] = m_bBox.m_vMin[0] + MyUnits<T>::angstrom() * (2 * uX + 1);
            atom.m_vPos[1] = m_bBox.m_vMin[1] + MyUnits<T>::angstrom() * (2 * uY + 1);
            atom.m_vPos[2] = m_bBox.m_vMin[2] + MyUnits<T>::angstrom() * (2 * uZ + 1);

            nvAssert(m_bBox.includes(atom.m_vPos)); // atom must be inside the bounding box
        }

        updateForces();
        m_fInitialPot = m_fCurPot;
    }

    struct Atom
    {
        NvU32 m_nProtons : 8;
        MyUnits<float> m_fCharge; // m_fCharge is a partial charge that appears because of proximity of other atoms
        rtvector<MyUnits<T>,3> m_vPos, m_vSpeed, m_vForce;
        MyUnits<double> m_fInaccuracy; // used as weight during energy equalization
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

        changeSpeedsToConserveEnery();
    }

    NvU32 getNNodes() const { return (NvU32)m_ocTree.size(); }
    OcTreeNode<Water>& accessNode(NvU32 index) { return m_ocTree[index]; }
    void resizeNodes(NvU32 nNodes) { m_ocTree.resize(nNodes); }

    // returns true if contributions between those two boxes are fully accounted for (either just now or before - at higher level of hierarchy)
    bool addNode2LeafContribution(NvU32 dstLeafIndex, const OcBoxStack<T>& dstLeafStack, NvU32 srcNodeIndex, const OcBoxStack<T>& srcNodeStack)
    {
        nvAssert(m_ocTree[dstLeafIndex].getNPoints() && m_ocTree[srcNodeIndex].getNPoints());
        // check if we can treat srcNode as one point as opposed to looking at its individual sub-boxes or points
        if (!dstLeafStack.isDescendent(srcNodeStack))
        {
            const auto& dstBox = setUnits<MyUnits<T>>(dstLeafStack.getCurBox());
            const auto& srcBox = setUnits<MyUnits<T>>(srcNodeStack.getCurBox());
            // if boxes are too far - particles can't affect each other - rule that interactions are accounted for
            for (NvU32 uDim = 0; uDim < 3; ++uDim)
            {
                if (dstBox.m_vMin[uDim] > srcBox.m_vMax[uDim] + BondsDataBase<T>::s_zeroForceDist ||
                    srcBox.m_vMin[uDim] > dstBox.m_vMax[uDim] + BondsDataBase<T>::s_zeroForceDist)
                {
#if ASSERT_ONLY_CODE
                    m_dbgNContributions += m_ocTree[dstLeafIndex].getNPoints() * m_ocTree[srcNodeIndex].getNPoints();
#endif
                    return true;
                }
            }
        }
        auto& srcNode = m_ocTree[srcNodeIndex];
        if (!srcNode.isLeaf())
            return false;
        // if srcNode is leaf, then go over all its points and compute point-to-point interactions. as optimization, we can apply those interaction
        // in symmetric way - applying both force and counter-force at the same time. to avoid double-counting interactions, we only do that when:
        if (dstLeafIndex <= srcNodeIndex)
        {
            auto& dstNode = m_ocTree[dstLeafIndex];
            nvAssert(dstNode.getNPoints());
            for (NvU32 uSrcPoint = srcNode.getFirstPoint(); uSrcPoint < srcNode.getEndPoint(); ++uSrcPoint)
            {
                auto& srcPoint = m_points[uSrcPoint];
                for (NvU32 uDstPoint = dstNode.getFirstPoint(); uDstPoint < dstNode.getEndPoint(); ++uDstPoint)
                {
                    if (dstLeafIndex == srcNodeIndex && uDstPoint >= uSrcPoint) // avoid double-counting
                        continue;
                    auto& dstPoint = m_points[uDstPoint];
                    auto& eBond = BondsDataBase<T>::getEBond(dstPoint.m_nProtons, srcPoint.m_nProtons, 1);
                    typename BondsDataBase<T>::LJ_Out out;
                    out.vForce = dstPoint.m_vPos - srcPoint.m_vPos;
                    for (NvU32 uDim = 0; uDim < 3; ++uDim) // particles positions must wrap around the boundary of bounding box
                    {
                        if (out.vForce[uDim] < -m_fHalfBoxSize) out.vForce[uDim] += m_fBoxSize;
                        else if (out.vForce[uDim] > m_fHalfBoxSize) out.vForce[uDim] -= m_fBoxSize;
                    }
                    if (eBond.lennardJones(out.vForce, out))
                    {
                        dstPoint.m_vForce += out.vForce;
                        srcPoint.m_vForce -= out.vForce;
                        m_fCurPot += out.fPotential;
                    }
#if ASSERT_ONLY_CODE
                    m_dbgNContributions += 2;
#endif
                }
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
    // the forces between atoms change rapidly depending on distance - so time discretization introduces significant errors into simulation. since we can't
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
            fTotalWeight += m_points[uPoint].m_fInaccuracy;
        }
        m_fCurKin.clear();
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            auto& point = m_points[uPoint];
            MyUnits<T> fCurWeight = point.m_fInaccuracy / fTotalWeight;
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
        OcBoxStack<T> stack(0, removeUnits(m_bBox));
        // split oc-tree recursively to get small number of points per leaf
        splitRecursive(0, stack);

        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            auto& point = m_points[uPoint];
            point.m_vForce.set(MyUnits<T>(0));
        }

#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#endif
        m_fCurPot.clear(); // prepare for potential energy computation
        m_ocTree[0].computeForces(0, removeUnits(m_bBox), *this);
        nvAssert(m_dbgNContributions == m_points.size() * (m_points.size() - 1));
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
                rtvector<MyUnits<T>, 3> vAcceleration = newtonLaw(BondsDataBase<T>::getAtom(point.m_nProtons).m_fMass, point.m_vForce);
                rtvector<MyUnits<T>, 3> vNewSpeed = point.m_vSpeed + vAcceleration * fHalfTimeStep;
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
            point.m_fInaccuracy = (point.m_fInaccuracy + lengthSquared(vAcceleration)) / 2;
            point.m_vSpeed += vAcceleration * fHalfTimeStep;
            if (VERLET_STEP_INDEX == 1)
            {
                m_fCurKin += lengthSquared(point.m_vSpeed) * fMass / 2;
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

    void splitRecursive(const NvU32 uNode, OcBoxStack<T> &stack)
    {
        auto* pNode = &m_ocTree[uNode];
        nvAssert(pNode->isLeaf());
        pNode->m_nodeData.m_fTotalCharge.clear();
        if (pNode->getNPoints() <= 16) // no need to split further?
        {
            // update total node charge
            for (NvU32 uPoint = pNode->getFirstPoint(); uPoint < pNode->getEndPoint(); ++uPoint)
            {
                auto& point = m_points[uPoint];
                pNode->m_nodeData.m_fTotalCharge += point.m_fCharge;
            }
            return;
        }
#if ASSERT_ONLY_CODE
        NvU32 dbgNPoints1 = pNode->getNPoints(), dbgNPoints2 = 0;
#endif
        pNode->split(stack, *this);
        NvU32 uFirstChild = m_ocTree[uNode].getFirstChild();
        for (NvU32 uChild = 0; uChild < 8; ++uChild)
        {
#if ASSERT_ONLY_CODE
            NvU32 dbgDepth = stack.getCurDepth();
            auto dbgBox = stack.getBox(dbgDepth);
#endif
            stack.push(uChild, uFirstChild + uChild);
#if ASSERT_ONLY_CODE
            nvAssert(stack.getCurDepth() == dbgDepth + 1);
            auto& node = m_ocTree[uFirstChild + uChild];
            dbgNPoints2 += node.getNPoints();
            if (stack.getCurDepth() == 1)
            {
                for (NvU32 u = node.getFirstPoint(); u < node.getEndPoint(); ++u)
                {
                    nvAssert(stack.getBox(stack.getCurDepth()).includes(removeUnits(m_points[u].m_vPos)));
                }
            }
#endif
            splitRecursive(uFirstChild + uChild, stack);
            m_ocTree[uNode].m_nodeData.m_fTotalCharge += m_ocTree[uFirstChild + uChild].m_nodeData.m_fTotalCharge;
#if ASSERT_ONLY_CODE
            NvU32 childIndex = stack.pop();
            nvAssert(childIndex == uChild);
            nvAssert(stack.getCurDepth() == dbgDepth);
            nvAssert(stack.getBox(dbgDepth) == dbgBox);
#endif
        }
        nvAssert(dbgNPoints1 == dbgNPoints2);
    }

    MyUnits<T> m_fBoxSize, m_fHalfBoxSize;
    BBox3<MyUnits<T>> m_bBox;
    std::vector<Atom> m_points;
    std::vector<OcTreeNode<Water>> m_ocTree;
    RNGUniform m_rng;

    MyUnits<T> m_fCurPot, m_fCurKin, m_fInitialPot; // energy conservation variables
    MyUnits<T> m_fTimeStep = MyUnits<T>::nanoSecond() * 0.000001;
    MyUnits<T> m_fMaxSpaceStep = MyUnits<T>::nanoMeter() / 40, m_fMaxSpaceStepSqr = m_fMaxSpaceStep * m_fMaxSpaceStep;

#if ASSERT_ONLY_CODE
    NvU64 m_dbgNContributions;
#endif
};