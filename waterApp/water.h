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
        Force(NvU32 uAtom1, NvU32 uAtom2) : m_uAtom1(uAtom1), m_uAtom2(uAtom2), m_collisionDetected(0)
        {
            nvAssert(m_uAtom1 == uAtom1 && m_uAtom2 == uAtom2 && m_uAtom1 != m_uAtom2);
        }
        NvU32 getAtom1Index() const { return m_uAtom1; }
        NvU32 getAtom2Index() const { return m_uAtom2; }
        bool hadCollision() const { return m_collisionDetected; }
        void notifyCollision() { m_collisionDetected = 1; }

        inline bool operator <(const Force& other) const
        {
            return m_fPotential[1] - m_fPotential[0] < other.m_fPotential[1] - other.m_fPotential[0];
        }

        MyUnits<T> m_fPotential[2]; // potentials corresponding to vPos[0] and vPos[1] respectively
        MyUnits<T> m_fDistSqr[2]; // distances between atoms corresponding to vPos[0] and vPos[1] respectively

    private:
        NvU32 m_uAtom1 : 15;
        NvU32 m_uAtom2 : 15;
        NvU32 m_collisionDetected : 1; // collision detected during time step
    };

    struct NODE_DATA // data that we store in each node
    {
    };

    Water() : m_ocTree(*this)
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
    }

    struct Atom
    {
        inline MyUnits<T> getMass() const { return BondsDataBase<T>::getAtom(m_nProtons).m_fMass; }
        NvU32 m_nProtons : 8;
        rtvector<MyUnits<T>,3> m_vPos[2], m_vSpeed[2], m_vForce;
    };
    inline std::vector<Atom>& points()
    {
        return m_points;
    }

    void makeTimeStep()
    {
        createListOfForces();

        for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
        {
            auto& atom = m_points[uAtom];
            atom.m_vForce = rtvector<MyUnits<T>, 3>();
        }
        for (NvU32 forceIndex = 0; forceIndex < m_forces.size(); ++forceIndex)
        {
            auto& force = m_forces[forceIndex];
            force.m_fPotential[0] = MyUnits<T>();
            updateForces<0>(force);
        }

        for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
        {
            auto& atom = m_points[uAtom];
            advectPosition(atom, m_fTimeStep);
            atom.m_vSpeed[1] = atom.m_vSpeed[0];
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

        // update m_vSpeed[1] in each atom based on change of force potentials
        for (NvU32 forceIndex = 0; forceIndex < m_forces.size(); ++forceIndex)
        {
            auto& force = m_forces[forceIndex];
            updateSpeeds(force);
        }

        // update kinetic energy
        m_fCurTotalKin = MyUnits<T>();
        for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
        {
            auto& atom = m_points[uAtom];
            MyUnits<T> fMass = atom.getMass();
            atom.m_vPos[0] = atom.m_vPos[1];
            atom.m_vSpeed[0] = atom.m_vSpeed[1];
            MyUnits<T> fKin = lengthSquared(atom.m_vSpeed[0]) * fMass / 2;
            m_fCurTotalKin += fKin;
        }

        // if the speeds get too high - scale them to achieve required average kinetic energy (and thus required avg. temp)
        auto fScaleCoeff = (m_fWantedAverageKin / (m_fCurTotalKin / (T)m_points.size())).m_value;
        if (fScaleCoeff < 1)
        {
            double fScaleCoeffSqrt = sqrt(fScaleCoeff);
            for (NvU32 uAtom = 0; uAtom < m_points.size(); ++uAtom)
            {
                auto& atom = m_points[uAtom];
                // kinetic energy is computed using the following equation:
                // fKin = lengthSquared(atom.m_vSpeed[0]) * fMass / 2;
                // hence if we multiply fKin by fScaleCoeff, we must multiply speed by sqrt(fScaleCoeff);
                atom.m_vSpeed[0] *= fScaleCoeffSqrt;
            }
            m_fCurTotalKin *= fScaleCoeff;
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

        force.notifyCollision();
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

    NvU32 getNNodes() const { return (NvU32)m_ocTree.m_nodes.size(); }
    OcTreeNode<Water>& accessNode(NvU32 index) { return m_ocTree.m_nodes[index]; }
    bool isOkToBeNotLeaf(const OcTreeNode<Water>& node) const
    {
        return node.getNPoints() > 0; // we would we split node with 0 points in it?
    }
    bool canHaveInteraction(const OcTreeNode<Water>& node) const
    {
        return node.getNPoints() > 0;
    }

    // returns true if after this call interaction between those two boxes are fully accounted for
    bool addLeafAndNodeInteraction(NvU32 leafIndex, const OcBoxStack<T>& leafStack, NvU32 nodeIndex, const OcBoxStack<T>& nodeStack)
    {
        nvAssert(m_ocTree.m_nodes[leafIndex].getNPoints() && m_ocTree.m_nodes[nodeIndex].getNPoints());
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
                    m_dbgNContributions += 2 * m_ocTree.m_nodes[leafIndex].getNPoints() * m_ocTree.m_nodes[nodeIndex].getNPoints();
#endif
                    return true;
                }
            }
            // we want to descend until leafs because it's possible some nodes will be cut off early that way
            auto& node = m_ocTree.m_nodes[nodeIndex];
            if (!node.isLeaf()) return false;
        }
        auto& leafNode1 = m_ocTree.m_nodes[leafIndex];
        nvAssert(leafNode1.getNPoints());
        auto& leafNode2 = m_ocTree.m_nodes[nodeIndex];
        for (NvU32 uTreePoint2 = leafNode2.getFirstTreePoint(); uTreePoint2 < leafNode2.getEndTreePoint(); ++uTreePoint2)
        {
            NvU32 uPoint2 = m_ocTree.getPointIndex(uTreePoint2);
            auto& point2 = m_points[uPoint2];
            for (NvU32 uTreePoint1 = (leafIndex == nodeIndex) ? uTreePoint2 + 1 : leafNode1.getFirstTreePoint(); uTreePoint1 < leafNode1.getEndTreePoint(); ++uTreePoint1)
            {
#if ASSERT_ONLY_CODE
                m_dbgNContributions += 2;
#endif
                NvU32 uPoint1 = m_ocTree.getPointIndex(uTreePoint1);
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
    rtvector<T, 3> getPointPos(const NvU32 index) const { return removeUnits(m_points[index].m_vPos[0]); }

private:
    void createListOfForces()
    {
        m_ocTree.rebuild(removeUnits(m_bBox), (NvU32)m_points.size());

#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#endif
        m_forces.resize(0);
        m_ocTree.m_nodes[0].addNode2NodeInteractions(0, removeUnits(m_bBox), *this);
        nvAssert(m_dbgNContributions == m_points.size() * (m_points.size() - 1));
    }

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
        NvU32 uAtom2 = force.getAtom2Index();
        auto& atom2 = m_points[uAtom2];

        rtvector<MyUnits<T>, 3> vR;
        typename BondsDataBase<T>::LJ_Out out;
        if (computeForce<index>(atom1, atom2, vR, out))
        {
            // symmetric addition should ensure conservation of momentum
            if (index == 0)
            {
                atom1.m_vForce += out.vForce;
                atom2.m_vForce -= out.vForce;
            }
            force.m_fPotential[index] = out.fPotential;
        }
        force.m_fDistSqr[index] = out.fDistSqr; // this is needed even if force is 0
    }
    void updateSpeeds(Force &force)
    {
        NvU32 uAtom1 = force.getAtom1Index();
        auto& atom1 = m_points[uAtom1];
        MyUnits<T> fMass1 = atom1.getMass();
        NvU32 uAtom2 = force.getAtom2Index();
        auto& atom2 = m_points[uAtom2];
        MyUnits<T> fMass2 = atom2.getMass();

        auto vDir = computeDir<1>(atom2, atom1);
        nvAssert(dot(vDir, vDir) == force.m_fDistSqr[1]);
        vDir /= sqrt(force.m_fDistSqr[1]);
        auto fV1 = dot(atom1.m_vSpeed[0], vDir);
        auto fV2 = dot(atom2.m_vSpeed[0], vDir);

        auto fCommonTerm = fMass1 * fMass2 * (fV1 - fV2);
        auto fSqrtTerm = (force.m_fPotential[0] - force.m_fPotential[1]) * 2 * (fMass1 + fMass2) + fMass1 * fMass2 * sqr(fV1 - fV2);
        // if that term is negative, this means atoms were moving against the force and their speed must have decreased, however
        // their speed was too small to start with and it can't decrease enough to compensate for increase in potential energy
        if (fSqrtTerm < 0)
        {
            fSqrtTerm = MyUnits<T>();
        }
        else
        {
            fSqrtTerm = sqrt(fSqrtTerm * fMass1 * fMass2);
        }
        // Equations for wolfram cloud:
        // E1:=m1/2*V1^2+m2/2*V2^2+fPotPrev== m1/2*(V1+dV1)^2+m2/2*(V2+dV2)^2+fPotNext (* conservation of energy *)
        // E2:=m1*V1+m2*V2==m1*(V1+dV1)+m2*(V2+dV2) (* conservation of momentum *)
        // E3:=FullSimplify[Solve[E1&&E2,{dV1,dV2}]]
        // This yields two solutions:
        // - solution1 is where final speeds are directed towards each other (atoms will be getting closer)
        // - solution2 is where final speeds are directed away from each other (atoms will be getting farther apart)
        double fSign = 1.; // corresponds to solution1
        // solution2 happens in either of two cases:
        // - atoms have just collided and thus must now start moving away from each other
        // - atoms were moving away from each other on the previous time step, and distance between them has again increased on this time step
        nvAssert(force.m_fDistSqr > 0);
        if (force.hadCollision() || (fV1 - fV2 < 0 && force.m_fDistSqr[1] > force.m_fDistSqr[0]))
        {
            fSign = -1; // corresponds to solution 2
        }
        auto fDeltaV1 = ( fSqrtTerm * fSign - fCommonTerm) / (fMass1 * (fMass1 + fMass2));
        auto fDeltaV2 = (-fSqrtTerm * fSign + fCommonTerm) / (fMass2 * (fMass1 + fMass2));
        atom1.m_vSpeed[1] += vDir * fDeltaV1;
        atom2.m_vSpeed[1] += vDir * fDeltaV2;
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

    void advectPosition(Atom &atom, MyUnits<T> fTimeStep)
    {
        MyUnits<T> fMass = atom.getMass();
        auto vAvgSpeed = atom.m_vSpeed[0] + atom.m_vForce * (fTimeStep / 2 / fMass);
        atom.m_vPos[1] = wrapThePos(atom.m_vPos[0] + vAvgSpeed * fTimeStep);
    }

    MyUnits<T> m_fBoxSize, m_fHalfBoxSize;
    BBox3<MyUnits<T>> m_bBox;
    std::vector<Atom> m_points;
    std::vector<Force> m_forces;
    OcTree<Water> m_ocTree;
    RNGSobol m_rng;

    const double m_fWantedTempC = 37;
    MyUnits<T> m_fWantedAverageKin;
    MyUnits<T> m_fCurTotalKin; // energy conservation variables
    MyUnits<T> m_fTimeStep = MyUnits<T>::nanoSecond() * 0.0000000005;
    MyUnits<T> m_fMaxSpaceStep = MyUnits<T>::nanoMeter() / 512, m_fMaxSpaceStepSqr = m_fMaxSpaceStep * m_fMaxSpaceStep;

#if ASSERT_ONLY_CODE
    NvU64 m_dbgNContributions = 0;
#endif
};