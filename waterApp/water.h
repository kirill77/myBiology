#pragma once

#include <algorithm>
#include "basics/bonds.h"
#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGSobol.h"
#include "MonteCarlo/distributions.h"

// class used to wrap coordinates and directions so that everything stays inside boundind box
template <class T>
struct BoxWrapper : public BBox3<MyUnits<T>>
{
    BoxWrapper(MyUnits<T> fBoxSide = MyUnits<T>(1.))
    {
        m_fBoxSize = fBoxSide;
        m_fHalfBoxSize = m_fBoxSize / 2.;
        this->m_vMin = makeVector<MyUnits<T>, 3>(-m_fHalfBoxSize);
        this->m_vMax = makeVector<MyUnits<T>, 3>( m_fHalfBoxSize);
    }
    // if the atom exits bounding box, it enters from the other side
    rtvector<MyUnits<T>, 3> wrapThePos(const rtvector<MyUnits<T>, 3>& vOldPos) const
    {
        auto vNewPos = vOldPos;
        for (NvU32 uDim = 0; uDim < 3; ++uDim)
        {
            if (vNewPos[uDim] < this->m_vMin[uDim])
            {
                auto fOvershoot = (this->m_vMin[uDim] - vNewPos[uDim]);
                int nBoxSizes = 1 + (int)removeUnits(fOvershoot / m_fBoxSize);
                vNewPos[uDim] += m_fBoxSize * nBoxSizes;
                nvAssert(this->m_vMin[uDim] <= vNewPos[uDim] && vNewPos[uDim] <= this->m_vMax[uDim]);
                continue;
            }
            if (vNewPos[uDim] > this->m_vMax[uDim])
            {
                auto fOvershoot = (vNewPos[uDim] - this->m_vMax[uDim]);
                int nBoxSizes = 1 + (int)removeUnits(fOvershoot / m_fBoxSize);
                vNewPos[uDim] -= m_fBoxSize * nBoxSizes;
                nvAssert(this->m_vMin[uDim] <= vNewPos[uDim] && vNewPos[uDim] <= this->m_vMax[uDim]);
            }
        }
        nvAssert(this->includes(vNewPos)); // atom must be inside the bounding box
        return vNewPos;
    }
    template <NvU32 index>
    rtvector<MyUnits<T>, 3> computeDir(const Atom<T>& atom1, const Atom<T>& atom2) const
    {
        rtvector<MyUnits<T>, 3> vOutDir = atom1.getUnwrappedPos(index) - atom2.getUnwrappedPos(index);
        for (NvU32 uDim = 0; uDim < 3; ++uDim) // particles positions must wrap around the boundary of bounding box
        {
            if (vOutDir[uDim] < -m_fHalfBoxSize) vOutDir[uDim] += m_fBoxSize;
            else if (vOutDir[uDim] > m_fHalfBoxSize) vOutDir[uDim] -= m_fBoxSize;
        }
        return vOutDir;
    }

private:
    MyUnits<T> m_fBoxSize, m_fHalfBoxSize;
};

template <class _T>
struct Water
{
    typedef _T T;

    typedef std::unordered_map<ForceKey, Force<T>> ForceMap;

    const ForceMap &getForces() const { return m_forces; }

    struct NODE_DATA // data that we store in each node
    {
    };

    Water() : m_ocTree(*this), m_bBox(MyUnits<T>::angstrom() * 20)
    {
        MyUnits<T> volume = m_bBox.evalVolume();
        // one mole of water has volume of 18 milliliters
        NvU32 nWaterMolecules = (NvU32)(AVOGADRO * volume.m_value / MyUnits<T>::milliLiter().m_value / 18);
#ifdef NDEBUG
        m_atoms.resize(3 * nWaterMolecules);
#else
        // debug can't simulate all molecules - too slow
        m_atoms.resize(3 * 64);
#endif

        NvU32 nOs = 0, nHs = 0;

        for (NvU32 u = 0; u < m_atoms.size(); ++u)
        {
            Atom<T> &atom = m_atoms[u];
            if (nHs < nOs * 2)
            {
                atom = Atom<T>(NPROTONS_H);
                ++nHs;
            }
            else
            {
                atom = Atom<T>(NPROTONS_O);
                ++nOs;
            }

            rtvector<MyUnits<T>, 3> vNewPos;
            for (NvU32 uDim = 0; uDim < 3; ++uDim)
            {
                double f = m_rng.generate01();
                vNewPos[uDim] = m_bBox.m_vMin[uDim] * f + m_bBox.m_vMax[uDim] * (1 - f);
            }
            atom.setPos(0, vNewPos, m_bBox);
            m_rng.nextSeed();

            if (!m_bBox.includes(atom.getUnwrappedPos(0))) // atom must be inside the bounding box
            {
                __debugbreak();
            }
        }

        m_fWantedAverageKin = MyUnits<T>::fromCelcius(m_fWantedTempC);
    }

    inline std::vector<Atom<T>>& points()
    {
        return m_atoms;
    }

    void makeTimeStep()
    {
        updateListOfForces();

        for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
        {
            auto& atom = m_atoms[uAtom];
            atom.m_vForce = rtvector<MyUnits<T>, 3>();
        }
        for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
        {
            Force<T> &force = _if->second;
            updateForces<0>(_if->first, force);
        }

        for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
        {
            // advect positions by full step and speeds by half-step
            auto& atom = m_atoms[uAtom];
            MyUnits<T> fMass = atom.getMass();
            atom.m_vSpeed[1] = atom.m_vSpeed[0] + atom.m_vForce * (m_fTimeStep / 2 / fMass);
            atom.setPos(1, atom.getUnwrappedPos(0) + atom.m_vSpeed[1] * m_fTimeStep, m_bBox);
            // clear forces before we start accumulating them for next step
            atom.m_vForce = rtvector<MyUnits<T>, 3>();
        }

        for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
        {
            Force<T> &force = _if->second;
            updateForces<1>(_if->first, force);
        }

        for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
        {
            // advect speeds by half-step
            auto& atom = m_atoms[uAtom];
            MyUnits<T> fMass = atom.getMass();
            atom.m_vSpeed[0] = atom.m_vSpeed[1] + atom.m_vForce * (m_fTimeStep / 2 / fMass);
        }

        // update kinetic energy
        m_fCurTotalKin = MyUnits<T>();
        for (NvU32 uAtom = 0; uAtom < m_atoms.size(); ++uAtom)
        {
            auto& atom = m_atoms[uAtom];
            MyUnits<T> fMass = atom.getMass();
            atom.copyPos(0, 1);
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

            if (force.dissociateWeakBond(forceKey, atom1, atom2, m_bBox))
            {
                _if = m_forces.erase(_if);
                if (_if == m_forces.end())
                    break;
            }
        }
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
                auto vDir = m_bBox.computeDir<0>(atom1, atom2);
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
    rtvector<T, 3> getPointPos(const NvU32 index) const { return removeUnits(m_atoms[index].getUnwrappedPos(0)); }
    rtvector<MyUnits<T>, 3> computeDir(const Atom<T>& atom1, const Atom<T>& atom2) const { return m_bBox.computeDir<0>(atom1, atom2); }

private:
    void updateListOfForces()
    {
        dissociateWeakBonds();

        m_ocTree.rebuild(removeUnits(m_bBox), (NvU32)m_atoms.size());

#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#endif

        m_ocTree.m_nodes[0].addNode2NodeInteractions(0, removeUnits(m_bBox), *this);
        nvAssert(m_dbgNContributions == m_atoms.size() * (m_atoms.size() - 1));
    }

    template <NvU32 index>
    void updateForces(ForceKey forceKey, Force<T> &force)
    {
        auto& atom1 = m_atoms[forceKey.getAtom1Index()];
        auto& atom2 = m_atoms[forceKey.getAtom2Index()];

        rtvector<MyUnits<T>, 3> vForce;
        if (force.computeForce<index>(atom1, atom2, m_bBox, vForce))
        {
            atom1.m_vForce += vForce;
            atom2.m_vForce -= vForce;
        }
    }

    BoxWrapper<T> m_bBox;
    std::vector<Atom<T>> m_atoms;
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