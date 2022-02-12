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

    typedef std::unordered_map<ForceKey, Force<T>> ForceMap;

    const ForceMap &getForces() const { return m_forces; }

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
        m_atoms.resize(3 * 64);
#endif

        NvU32 nOs = 0, nHs = 0;

        for (NvU32 u = 0; u < m_atoms.size(); ++u)
        {
            Atom& atom = m_atoms[u];
            if (nHs < nOs * 2)
            {
                atom = Atom(NPROTONS_H);
                ++nHs;
            }
            else
            {
                atom = Atom(NPROTONS_O);
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
        Atom(NvU32 nProtons = 1) : m_nBondedAtoms(0), m_nProtons(nProtons), m_uValence(BondsDataBase<T>::getAtom(m_nProtons).m_uValence)
        {
            for (NvU32 u = 0; u < m_bondedAtoms.size(); ++u) m_bondedAtoms[u] = -1;
            nvAssert(m_uValence != 0); // we don't work with noble gasses
        }

        NvU32 getNProtons() const { return m_nProtons; }
        MyUnits<T> getMass() const { return BondsDataBase<T>::getAtom(m_nProtons).m_fMass; }
        NvU32 getValence() const { return m_uValence; }
        NvU32 getNBonds() const { return m_nBondedAtoms; }
        NvU32 getBond(NvU32 uBond) const { nvAssert(uBond < m_nBondedAtoms); return m_bondedAtoms[uBond]; }

        void addBond(NvU32 uAtom)
        {
            nvAssert(m_nBondedAtoms < m_uValence);
            m_bondedAtoms[m_nBondedAtoms] = uAtom;
            nvAssert(m_bondedAtoms[m_nBondedAtoms] == uAtom); // check that type conversion didn't loose information
            ++m_nBondedAtoms;
        }
        void removeBond(NvU32 uAtom)
        {
            for (NvU32 u = 0; ; ++u)
            {
                nvAssert(u < m_nBondedAtoms);
                if (m_bondedAtoms[u] == uAtom)
                {
                    m_bondedAtoms[u] = m_bondedAtoms[--m_nBondedAtoms];
                    return;
                }
            }
        }

        rtvector<MyUnits<T>,3> m_vPos[2], m_vSpeed[2], m_vForce;

        private:
            union
            {
                NvU32 flags;
                struct
                {
                    NvU32 m_nProtons : 8;
                    NvU32 m_nBondedAtoms : 3;
                    NvU32 m_uValence : 3;
                };
            };
            std::array<unsigned short, 4> m_bondedAtoms;
    };
    inline std::vector<Atom>& points()
    {
        return m_atoms;
    }

    void makeTimeStep()
    {
        createListOfForces();

        for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
        {
            auto& atom = m_atoms[uAtom];
            atom.m_vForce = rtvector<MyUnits<T>, 3>();
        }
        for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
        {
            Force<T> &force = _if->second;
            force.m_fPotential[0] = MyUnits<T>();
            updateForces<0>(_if->first, force);
        }

        for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
        {
            auto& atom = m_atoms[uAtom];
            advectPosition(atom, m_fTimeStep);
            atom.m_vSpeed[1] = atom.m_vSpeed[0];
        }

        // don't let atoms come closer than the bond length - otherwise enormous repulsive force explodes the simulation
        for ( ; ; )
        {
            int nAdjustments = 0;
            for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
            {
                nAdjustments += adjustForceDistance(_if->first, _if->second);
            }
            // if nothing has been adjusted - break
            if (nAdjustments == 0)
            {
                break;
            }
        }

        dissociateWeakBonds();

        for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
        {
            Force<T> &force = _if->second;
            force.m_fPotential[1] = MyUnits<T>();
            updateForces<1>(_if->first, force);
        }

        // update m_vSpeed[1] in each atom based on change of force potentials
        for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
        {
            updateSpeeds(_if->first, _if->second);
        }

        // update kinetic energy
        m_fCurTotalKin = MyUnits<T>();
        for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
        {
            auto& atom = m_atoms[uAtom];
            MyUnits<T> fMass = atom.getMass();
            atom.m_vPos[0] = atom.m_vPos[1];
            atom.m_vSpeed[0] = atom.m_vSpeed[1];
            MyUnits<T> fKin = lengthSquared(atom.m_vSpeed[0]) * fMass / 2;
            m_fCurTotalKin += fKin;
        }

        // if the speeds get too high - scale them to achieve required average kinetic energy (and thus required avg. temp)
        auto fScaleCoeff = (m_fWantedAverageKin / (m_fCurTotalKin / (T)m_atoms.size())).m_value;
        if (fScaleCoeff < 1)
        {
            double fScaleCoeffSqrt = sqrt(fScaleCoeff);
            for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
            {
                auto& atom = m_atoms[uAtom];
                // kinetic energy is computed using the following equation:
                // fKin = lengthSquared(atom.m_vSpeed[0]) * fMass / 2;
                // hence if we multiply fKin by fScaleCoeff, we must multiply speed by sqrt(fScaleCoeff);
                atom.m_vSpeed[0] *= fScaleCoeffSqrt;
            }
            m_fCurTotalKin *= fScaleCoeff;
        }
    }

    void dissociateWeakBonds()
    {
        for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
        {
            Force<T> &force = _if->second;

            // compute current bond length
            auto& forceKey = _if->first;
            auto &atom1 = m_atoms[forceKey.getAtom1Index()];
            auto &atom2 = m_atoms[forceKey.getAtom2Index()];
            auto vDir = computeDir<1>(atom1, atom2);
            auto fDistSqr = dot(vDir, vDir);

            auto& bond = BondsDataBase<T>::getEBond(atom1.getNProtons(), atom2.getNProtons(), 1);
            // if atoms are too far apart - erase the force
            if (fDistSqr >= BondsDataBase<T>::s_zeroForceDistSqr)
            {
                _if = m_forces.erase(_if);
                if (_if == m_forces.end())
                    break;
            }
            // check covalent bond threshold - it's smaller than global zero-force threshold
            else if (force.isCovalentBond() && fDistSqr >= bond.m_fDissocLengthSqr)
            {
                force.dropCovalentBond();
                atom1.removeBond(forceKey.getAtom2Index());
                atom2.removeBond(forceKey.getAtom1Index());
            }
        }
    }

    int adjustForceDistance(ForceKey forceKey, Force<T> &force)
    {
        NvU32 uAtom1 = forceKey.getAtom1Index();
        auto& atom1 = m_atoms[uAtom1];
        NvU32 uAtom2 = forceKey.getAtom2Index();
        auto& atom2 = m_atoms[uAtom2];
        auto& eBond = BondsDataBase<T>::getEBond(atom1.getNProtons(), atom2.getNProtons(), 1);
        auto vDir = computeDir<1>(atom1, atom2);
        auto fDistSqr = lengthSquared(vDir);
        // is distance between the atoms larger than the bonth length? then we don't have to do anything
        if (fDistSqr > eBond.m_fLengthSqr)
            return 0;

        force.notifyCollision();

        // if this force is not yet covalent bond and atoms have vacant orbitals - we make this force a covalent bond here
        if (!force.isCovalentBond() && atom1.getNBonds() < atom1.getValence() && atom2.getNBonds() < atom2.getValence())
        {
            atom1.addBond(uAtom2);
            atom2.addBond(uAtom1);
            force.setCovalentBond();
        }

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
            auto& atom2 = m_atoms[uPoint2];
            for (NvU32 uTreePoint1 = (leafIndex == nodeIndex) ? uTreePoint2 + 1 : leafNode1.getFirstTreePoint(); uTreePoint1 < leafNode1.getEndTreePoint(); ++uTreePoint1)
            {
#if ASSERT_ONLY_CODE
                m_dbgNContributions += 2;
#endif
                NvU32 uPoint1 = m_ocTree.getPointIndex(uTreePoint1);
                auto& atom1 = m_atoms[uPoint1];
                auto vDir = computeDir<0>(atom2, atom1);
                auto fLengthSqr = lengthSquared(vDir);
                if (fLengthSqr >= BondsDataBase<T>::s_zeroForceDistSqr) // if atoms are too far away - disregard
                {
                    continue;
                }

                for (NvU32 uBond1 = 0; ; ++uBond1)
                {
                    if (uBond1 >= atom1.getNBonds())
                    {
                        m_forces[ForceKey(uPoint1, uPoint2)]; // this adds default force between those two atoms into ForceMap
                        break;
                    }
                    // if this is covalent bond - we don't need to add it - it must already be in the list of forces
                    if (atom1.getBond(uBond1) == uPoint2)
                    {
                        break;
                    }
                }
            }
        }
        return true;
    }

    const BBox3<MyUnits<T>>& getBoundingBox() const { return m_bBox; }

    MyUnits<T> evalTemperature() const
    {
        return MyUnits<T>::evalTemperature(m_fCurTotalKin / (NvU32)m_atoms.size());
    }
    MyUnits<T> evalPressure() const
    {
        return MyUnits<T>::evalPressure(m_fCurTotalKin, m_bBox.evalVolume(), (NvU32)m_atoms.size());
    }
    const MyUnits<T> &getCurTimeStep() const { return m_fTimeStep; }
    rtvector<T, 3> getPointPos(const NvU32 index) const { return removeUnits(m_atoms[index].m_vPos[0]); }

private:
    void createListOfForces()
    {
        m_ocTree.rebuild(removeUnits(m_bBox), (NvU32)m_atoms.size());

#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#endif

        m_ocTree.m_nodes[0].addNode2NodeInteractions(0, removeUnits(m_bBox), *this);
        nvAssert(m_dbgNContributions == m_atoms.size() * (m_atoms.size() - 1));
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
    void updateForces(ForceKey forceKey, Force<T> &force)
    {
        NvU32 uAtom1 = forceKey.getAtom1Index();
        auto& atom1 = m_atoms[uAtom1];
        NvU32 uAtom2 = forceKey.getAtom2Index();
        auto& atom2 = m_atoms[uAtom2];

        rtvector<MyUnits<T>, 3> vR = computeDir<index>(atom1, atom2);
        rtvector<MyUnits<T>, 3> vForce;
        if (force.computeForce<index>(atom1.getNProtons(), atom2.getNProtons(), vR, vForce) && index == 0)
        {
            atom1.m_vForce += vForce;
            atom2.m_vForce -= vForce;
        }
    }
    void updateSpeeds(ForceKey forceKey, Force<T> &force)
    {
        NvU32 uAtom1 = forceKey.getAtom1Index();
        auto& atom1 = m_atoms[uAtom1];
        MyUnits<T> fMass1 = atom1.getMass();
        NvU32 uAtom2 = forceKey.getAtom2Index();
        auto& atom2 = m_atoms[uAtom2];
        MyUnits<T> fMass2 = atom2.getMass();

        auto vDir = computeDir<1>(atom2, atom1);
        nvAssert(dot(vDir, vDir) == force.m_fDistSqr[1]);
        vDir /= sqrt(force.m_fDistSqr[1]);
        auto fV1 = dot(atom1.m_vSpeed[0], vDir);
        auto fV2 = dot(atom2.m_vSpeed[0], vDir);

        // if this is covalent bond and atoms had collision - we expect atoms to stick together (inelastic collission)
        if (force.isCovalentBond() && force.hadCollision())
        {
            auto fV = (fV1 * fMass1 + fV2 * fMass2) / (fMass1 + fMass2);
            atom1.m_vSpeed[1] += vDir * (fV - fV1);
            atom2.m_vSpeed[2] += vDir * (fV - fV2);
            return;
        }

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
        // E1:=m1/2*V1^2+m2/2*V2^2+fPotPrev==m1/2*(V1+dV1)^2+m2/2*(V2+dV2)^2+fPotNext (* conservation of energy *)
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
    std::vector<Atom> m_atoms;
    ForceMap m_forces;
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