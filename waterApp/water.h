#pragma once

#include <algorithm>
#include <memory>
#include "basics/bonds.h"
#include "basics/myFilter.h"
#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGSobol.h"
#include "MonteCarlo/distributions.h"

template <class T>
using ForceMap = std::unordered_map<ForceKey, Force<T>>;

template <class T>
struct GlobalState
{
    GlobalState()
    {
        m_fWantedTempC = 37;
        m_fWantedAvgKin = MyUnits<T>::fromCelcius(m_fWantedTempC);
        // let's take a hydrogen atom having average kinetic energy
        // mHydrogen * V^2 / 2 = m_fDerivedAverageKin
        // then the V would be equal to sqrt(2 * m_fDerivedAverageKin) / mHydrogen
        // let's limit the speed to 100 times that
        auto hMass = Atom<T>(NPROTONS_H).getMass();
        nvAssert(hMass > 0 && hMass < 1e10);
        m_fMaxAllowedAtomSpeed = sqrt(m_fWantedAvgKin * 2) * 100 / hMass;
    }
    MyUnits<T> getWantedAvgKin() const { return m_fWantedAvgKin; }
    MyUnits<T> getWeightedKinSum() const { return m_fDoubleWeightedKinSum / 2; }
    MyUnits<T> getKinWeightsSum() const { return m_fKinWeightsSum; }
    void fastSpeedClamp(MyUnits3<T>& vSpeed) const
    {
        vSpeed[0] = std::clamp(vSpeed[0], -m_fMaxAllowedAtomSpeed, m_fMaxAllowedAtomSpeed);
        vSpeed[1] = std::clamp(vSpeed[1], -m_fMaxAllowedAtomSpeed, m_fMaxAllowedAtomSpeed);
        vSpeed[2] = std::clamp(vSpeed[2], -m_fMaxAllowedAtomSpeed, m_fMaxAllowedAtomSpeed);
    }

    void resetKinComputation()
    {
        m_fDoubleWeightedKinSum = MyUnits<T>();
        m_fKinWeightsSum = MyUnits<T>();
    }
    void notifyAtomSpeed(MyUnits<T> fMass, const MyUnits3<T>& vSpeed, MyUnits<T> fTimeStep)
    {
        m_fDoubleWeightedKinSum += fMass * lengthSquared(vSpeed) * fTimeStep;
        m_fKinWeightsSum += fTimeStep;
    }

private:
    T m_fWantedTempC;
    MyUnits<T> m_fWantedAvgKin, m_fMaxAllowedAtomSpeed;
    MyUnits<T> m_fDoubleWeightedKinSum, m_fKinWeightsSum;
};

template <class T>
struct SpeedScaler
{
    void scale(MyUnits<T> fInstantaneousAverageKin, MyUnits<T> fFilteredAverageKin, std::vector<Atom<T>>& atoms)
    {
        // this is to avoid situation when the average kinetic energy already got below target value,
        // but filtered value (being delayed due to filtering) didn't yet catch up
        if (fInstantaneousAverageKin < m_globalState.getWantedAvgKin())
            return;
        if (fFilteredAverageKin < m_globalState.getWantedAvgKin())
            return;
        // if the speeds get too high - scale them to achieve required average kinetic energy (and thus required avg. temp)
        double fScaleCoeff = (m_globalState.getWantedAvgKin() / fFilteredAverageKin).m_value;
        double fScaleCoeffSqrt = sqrt(fScaleCoeff);
        MyUnits<T> fSmallKinThreshold = m_globalState.getWantedAvgKin() / 2;
        for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
        {
            auto& atom = atoms[uAtom];
            // don't slow down the atoms that are already slow
            if (atom.getMass() * lengthSquared(atom.m_vSpeed) < fSmallKinThreshold)
                continue;
            // kinetic energy is computed using the following equation:
            // fKin = lengthSquared(atom.m_vSpeed) * fMass / 2;
            // hence if we want to multiply fKin by fScaleCoeff, we must multiply speed by sqrt(fScaleCoeff);
            atom.m_vSpeed *= fScaleCoeffSqrt;
        }
    }

private:
    GlobalState<T> m_globalState;
};

// class used to wrap coordinates and directions so that everything stays inside the simulation boundind box
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
    MyUnits3<T> wrapThePos(const MyUnits3<T>& vPos) const
    {
        MyUnits3<T> vNewPos = vPos;
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
    rtvector<MyUnits<T>, 3> computeDir(const Atom<T>& atom1, const Atom<T>& atom2) const
    {
        rtvector<MyUnits<T>, 3> vOutDir = atom1.m_vPos - atom2.m_vPos;
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

// * SimLayer provides hierarchy of diminishing timesteps for select atoms that require high accuracy (I call these the Detail atoms)
// * Each successive SimLayer divides time step by 2x
// * Detail atoms can be affected by either detail or non-detail atoms (I call latter the Proxy atoms)
// * Proxy atoms don't need to be simulated in detail - I assume they just travel on straight line with constant speed
// * To compute the path of a proxy atom, I have two positions (Atom<T>::m_vPos and Propagator<T>::m_vPos) and two times (see Propagator<T>::m_f*Time)

// propagator requires additional data per atom
template <class T>
struct PropagatorAtom
{
    void prepareForPropagation(const Atom<T>& atom, MyUnits<T> fBeginTime, MyUnits<T> fDbgEndTime)
    {
        m_vBeginSpeed = atom.m_vSpeed;
        m_vBeginPos = atom.m_vPos;
        m_fBeginTime = fBeginTime;
#if ASSERT_ONLY_CODE
        m_fDbgEndTime = fDbgEndTime;
#endif
        m_vEndForce = MyUnits3<T>();
    }
    void notifyStepShrinking(Atom<T>& atom, MyUnits<T> fBeginTime, MyUnits<T> fDbgEndTime)
    {
        atom.m_vPos = m_vBeginPos;
        atom.m_vSpeed = m_vBeginSpeed;
        nvAssert(m_fBeginTime == fBeginTime && fDbgEndTime < m_fDbgEndTime);
#if ASSERT_ONLY_CODE
        m_fDbgEndTime = fDbgEndTime;
#endif
    }
    void setInterpolatedPosition(Atom<T> &atom, MyUnits<T> fTime, const BoxWrapper<T>& w)
    {
        nvAssert(fTime >= m_fBeginTime && fTime <= m_fDbgEndTime);
        MyUnits<T> fTimeStep = fTime - m_fBeginTime;
        // advect positions by full step and speeds by half-step
        MyUnits3<T> vSpeed = m_vBeginSpeed + m_vBeginForce * (fTimeStep / 2 / atom.getMass());
        atom.m_vPos = w.wrapThePos(m_vBeginPos + vSpeed * fTimeStep);
    }
    void verletStep1(Atom<T>& atom, MyUnits<T> fTimeStep, const BoxWrapper<T>& w)
    {
        m_vBeginForce = m_vEndForce;
        // advect positions by full step and speeds by half-step
        atom.m_vSpeed += m_vBeginForce * (fTimeStep / 2 / atom.getMass());
        atom.m_vPos = w.wrapThePos(m_vBeginPos + atom.m_vSpeed * fTimeStep);
        // clear forces before we start accumulating them for next step
        m_vEndForce = MyUnits3<T>();
    }
    void verletStep2(Atom<T>& atom, MyUnits<T> fTimeStep)
    {
        atom.m_vSpeed += m_vEndForce * (fTimeStep / 2 / atom.getMass());
    }
    void notifyForceContribution(const MyUnits3<T>& vDeltaForce)
    {
        m_vEndForce += vDeltaForce;
    }

private:
    MyUnits<T> m_fBeginTime;
#if ASSERT_ONLY_CODE
    MyUnits<T> m_fDbgEndTime;
#endif
    MyUnits3<T> m_vBeginPos, m_vBeginSpeed, m_vBeginForce, m_vEndForce;
};
// force representation for SimLayer - unlike Propagator<T>, it stores all forces in linear array
template <class T>
struct SimLayerForce
{
    SimLayerForce() : m_forceKey(-1, -1) { }
    SimLayerForce(typename ForceMap<T>::const_local_iterator &it) : m_forceKey(it->first), m_force(it->second) { }
    SimLayerForce(typename std::vector<SimLayerForce<T>>::const_iterator& it) : m_forceKey(it->m_forceKey), m_force(it->m_force) { }
    ForceKey m_forceKey;
    Force<T> m_force;
    T m_fNormalizedForce0;
};
// used for interpolating of proxy atom positions and restoring them at the end
template <class T>
struct AtomPositionInterpolator
{
    AtomPositionInterpolator(bool isDetailAtom, Atom<T>& atom, PropagatorAtom<T>& prAtom, MyUnits<T> fTime, const BoxWrapper<T>& w) : m_atom(atom)
    {
        m_vPosBackup = atom.m_vPos;
        if (isDetailAtom)
        {
            return;
        }
        prAtom.setInterpolatedPosition(atom, fTime, w);
    }
    ~AtomPositionInterpolator()
    {
        m_atom.m_vPos = m_vPosBackup; // restore the original position
    }

private:
    Atom<T>& m_atom;
    MyUnits3<T> m_vPosBackup;
};
// simulation layer - hierarchy of ever diminishing time steps used for accuracy
template <class T, bool isTopLayer = false>
struct SimLayer
{
    SimLayer(NvU32 uLevel = 0) : m_uLevel(uLevel)
    {
        nvAssert(isTopLayer == (uLevel == 0));
        nvAssert(m_uLevel < 20);
    }
    void init(NvU32 nAtoms, const BoxWrapper<T> &bBox)
    {
        m_nAtoms = nAtoms;
        if (isTopLayer)
        {
            resetKinComputation();
        }
        else
        {
            m_atomsBitfield.assign((nAtoms + 31) / 32, 0);
            m_atomIndices.resize(0);
        }
        m_slForces.resize(0);
        m_slBox = bBox;
    }

    NvU32 numDetailAtoms() const { return isTopLayer ? m_nAtoms : (NvU32)m_atomIndices.size(); }
    bool isDetailAtom(NvU32 index) const { return isTopLayer ? true : m_atomsBitfield[index / 32] & (1 << (index & 31)); }
    void resetKinComputation()
    {
        m_globalState.resetKinComputation();
        if (m_pNextSimLayer) m_pNextSimLayer->resetKinComputation();
    }
    MyUnits<T> getWeightedKinSum() const
    {
        if (numDetailAtoms() == 0)
            return MyUnits<T>();
        return m_globalState.getWeightedKinSum() + m_pNextSimLayer->getWeightedKinSum();
    }
    MyUnits<T> getKinWeightsSum() const
    {
        if (numDetailAtoms() == 0)
            return MyUnits<T>();
        return m_globalState.getKinWeightsSum() + m_pNextSimLayer->getKinWeightsSum();
    }

    void addDetailAtom(NvU32 index)
    {
        nvAssert(!isTopLayer);
        if (isDetailAtom(index)) return; // don't add the same atom twice
        m_atomsBitfield[index / 32] |= (1 << (index & 31));
        NvU16 index16 = (NvU16)index;
        nvAssert(index16 == index); // check that no overflow happenned
        m_atomIndices.push_back(index);
    }

    template <class ForceArray>
    void prepareForPropagation(const ForceArray &forces, std::vector<Atom<T>>& atoms, std::vector<PropagatorAtom<T>>& prAtoms, MyUnits<T> fBeginTime, MyUnits<T> fDbgEndTime)
    {
        if (numDetailAtoms() == 0)
            return;
        // prepare list of forces that affect our atoms
        for (auto _if = forces.begin(); _if != forces.end(); ++_if)
        {
            SimLayerForce<T> slForce(_if);
            if (isDetailAtom(slForce.m_forceKey.getAtom1Index()) || isDetailAtom(slForce.m_forceKey.getAtom2Index()))
            {
                m_slForces.push_back(slForce);
            }
        }
        if (!isTopLayer)
        {
            // we have to reset all atoms we're going to re-simulate to their initial state
            for (NvU32 u = 0; u < m_atomIndices.size(); ++u)
            {
                NvU32 uAtom = m_atomIndices[u];
                Atom<T>& atom = atoms[uAtom];
                PropagatorAtom<T>& prAtom = prAtoms[uAtom];
                prAtom.notifyStepShrinking(atom, fBeginTime, fDbgEndTime);
            }
        }
    }
    void propagate(MyUnits<T> fBeginTime, MyUnits<T> fEndTime, std::vector<Atom<T>> &atoms, std::vector<PropagatorAtom<T>> &prAtoms)
    {
        if (numDetailAtoms() == 0)
            return;
        if (m_pNextSimLayer == nullptr)
        {
            m_pNextSimLayer = std::make_unique<SimLayer<T>>(m_uLevel + 1);
        }

        for (NvU32 u = 0; u < numDetailAtoms(); ++u)
        {
            NvU32 uAtom = isTopLayer ? u : m_atomIndices[u];
            PropagatorAtom<T>& prAtom = prAtoms[uAtom];
            prAtom.prepareForPropagation(atoms[uAtom], fBeginTime, fEndTime);
        }

        auto& nextSimLayer = *m_pNextSimLayer;
        updateForces<0>(fBeginTime, atoms, prAtoms, nextSimLayer);

        MyUnits<T> fTimeStep = fEndTime - fBeginTime;
        for (NvU32 u = 0; u < numDetailAtoms(); ++u)
        {
            NvU32 uAtom = isTopLayer ? u : m_atomIndices[u];
            Atom<T>& atom = atoms[uAtom];
            PropagatorAtom<T>& prAtom = prAtoms[uAtom];
            prAtom.verletStep1(atom, fTimeStep, m_slBox);
        }

        nextSimLayer.init((NvU32)atoms.size(), m_slBox);
        updateForces<1>(fEndTime, atoms, prAtoms, nextSimLayer);

        if (nextSimLayer.numDetailAtoms() < numDetailAtoms()) // do we have some detailed atoms of our own?
        {
            for (NvU32 u = 0; u < numDetailAtoms(); ++u)
            {
                NvU32 uAtom = isTopLayer ? u : m_atomIndices[u];
                if (nextSimLayer.isDetailAtom(uAtom))
                    continue;
                Atom<T>& atom = atoms[uAtom];
                PropagatorAtom<T>& prAtom = prAtoms[uAtom];
                MyUnits<T> fMass = atom.getMass();
                // this is the actual speed that was used to propel this atom on this time step
                m_globalState.notifyAtomSpeed(fMass, atom.m_vSpeed, fTimeStep);
                prAtom.verletStep2(atom, fTimeStep);
            }
        }

        if (nextSimLayer.numDetailAtoms() == 0)
            return;

        MyUnits<T> fMidTime = (fBeginTime + fEndTime) / 2;
        nextSimLayer.prepareForPropagation(m_slForces, atoms, prAtoms, fBeginTime, fMidTime);
        nextSimLayer.propagate(fBeginTime, fMidTime, atoms, prAtoms);
        nextSimLayer.propagate(fMidTime, fEndTime, atoms, prAtoms);
    }

private:
    template <NvU32 uPass> // pass 0 is first Verlet step, pass 1 is second Verlet step
    void updateForces(MyUnits<T> fTime, std::vector<Atom<T>>& atoms, std::vector<PropagatorAtom<T>>& prAtoms, SimLayer<T> &nextSimLayer)
    {
        for (NvU32 uF = 0; uF < m_slForces.size(); ++uF)
        {
            SimLayerForce<T>& slForce = m_slForces[uF];
            ForceKey forceKey = slForce.m_forceKey;
            Force<T>& force = slForce.m_force;

            NvU32 uAtom1 = forceKey.getAtom1Index();
            Atom<T>& atom1 = atoms[uAtom1];
            PropagatorAtom<T>& prAtom1 = prAtoms[uAtom1];
            AtomPositionInterpolator<T> interp1(isDetailAtom(uAtom1), atom1, prAtom1, fTime, m_slBox);
            NvU32 uAtom2 = forceKey.getAtom2Index();
            Atom<T>& atom2 = atoms[uAtom2];
            PropagatorAtom<T>& prAtom2 = prAtoms[uAtom2];
            AtomPositionInterpolator<T> interp2(isDetailAtom(uAtom2), atom2, prAtom2, fTime, m_slBox);

            // why would we simulate this force if it doesn't have any detail atoms in it?
            nvAssert(isDetailAtom(uAtom1) || isDetailAtom(uAtom2));

            ForceData<T> forceData;
            if (force.computeForce(atom1, atom2, m_slBox, forceData))
            {
                prAtom1.notifyForceContribution( forceData.vForce);
                prAtom2.notifyForceContribution(-forceData.vForce);
            }
            else
            {
                forceData.fNormalizedForce = 0;
            }
            if (uPass == 0)
            {
                slForce.m_fNormalizedForce0 = forceData.fNormalizedForce;
            }
            // if the normalized force has changed more than the threshold - need to simulate in more detail
            else if (fabs(forceData.fNormalizedForce - slForce.m_fNormalizedForce0) > 0.4)
            {
                if (isDetailAtom(uAtom1))
                {
                    nextSimLayer.addDetailAtom(uAtom1);
                }
                if (isDetailAtom(uAtom2))
                {
                    nextSimLayer.addDetailAtom(uAtom2);
                }
            }
        }
    }
    NvU32 m_nAtoms = 0;
    std::vector<NvU32> m_atomsBitfield;
    std::vector<NvU16> m_atomIndices;
    std::vector<SimLayerForce<T>> m_slForces;
    std::unique_ptr<SimLayer<T>> m_pNextSimLayer;
    BoxWrapper<T> m_slBox;
    NvU32 m_uLevel = 0;
    GlobalState<T> m_globalState;
};

// class propagates simulation by the given time step. if I detect that the time step is not small enough for some particular atom, this
// atom is passed for simulation to SimLayer class.
template <class T>
struct Propagator
{
    Propagator() : m_bBox(MyUnits<T>::angstrom() * 20) { }

    const ForceMap<T>& getForces() const { return m_forces; }
    inline const std::vector<Atom<T>>& points() const { return m_atoms; }
    const MyUnits<T>& getCurTimeStep() const { return m_fTimeStep; }
    rtvector<T, 3> getPointPos(const NvU32 index) const { return removeUnits(m_atoms[index].m_vPos); }
    rtvector<MyUnits<T>, 3> computeDir(const Atom<T>& atom1, const Atom<T>& atom2) const { return m_bBox.computeDir(atom1, atom2); }
    const BBox3<MyUnits<T>>& getBoundingBox() const { return m_bBox; }

    void propagate()
    {
        m_prAtoms.resize(m_atoms.size());
        m_topSimLayer.init((NvU32)m_atoms.size(), m_bBox);
        m_topSimLayer.prepareForPropagation(m_forces, m_atoms, m_prAtoms, MyUnits<T>(), m_fTimeStep);
        m_topSimLayer.propagate(MyUnits<T>(), m_fTimeStep, m_atoms, m_prAtoms);
    }

    void dissociateWeakBonds()
    {
        for (auto _if = m_forces.begin(); _if != m_forces.end(); ++_if)
        {
            Force<T>& force = _if->second;

            // compute current bond length
            auto& forceKey = _if->first;
            auto& atom1 = m_atoms[forceKey.getAtom1Index()];
            auto& atom2 = m_atoms[forceKey.getAtom2Index()];

            if (force.dissociateWeakBond(forceKey, atom1, atom2, m_bBox))
            {
                _if = m_forces.erase(_if);
                if (_if == m_forces.end())
                    break;
            }
        }
    }

protected:
    MyUnits<T> getInstantaneousAverageKin() const
    {
        MyUnits<T> fWeightsSum = m_topSimLayer.getKinWeightsSum();
        if (fWeightsSum == 0) return MyUnits<T>(); // to avoid division by 0
        MyUnits<T> fWeightedKinSum = m_topSimLayer.getWeightedKinSum();
        return fWeightedKinSum / fWeightsSum;
    }
    std::vector<Atom<T>> m_atoms;
    ForceMap<T> m_forces;
    BoxWrapper<T> m_bBox;

private:
    MyUnits<T> m_fTimeStep = MyUnits<T>::nanoSecond() * 0.0000000005;
    std::vector<PropagatorAtom<T>> m_prAtoms;
    SimLayer<T, true> m_topSimLayer; // top simulation layer
};

template <class _T>
struct Water : public Propagator<_T>
{
    typedef _T T;

    struct NODE_DATA // data that we store in each node
    {
    };

    Water() : m_ocTree(*this)
    {
        MyUnits<T> volume = this->m_bBox.evalVolume();
        // one mole of water has volume of 18 milliliters
        NvU32 nWaterMolecules = (NvU32)(AVOGADRO * volume.m_value / MyUnits<T>::milliLiter().m_value / 18);

#ifdef NDEBUG
        this->m_atoms.resize(3 * nWaterMolecules);
#else
        // debug can't simulate all molecules - too slow
        this->m_atoms.resize(3 * 64);
#endif

        for (NvU32 u = 0, nOs = 0, nHs = 0; u < this->m_atoms.size(); ++u)
        {
            Atom<T> &atom = this->m_atoms[u];
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
                vNewPos[uDim] = this->m_bBox.m_vMin[uDim] * f + this->m_bBox.m_vMax[uDim] * (1 - f);
            }
            atom.m_vPos = vNewPos;
            m_rng.nextSeed();

            if (!this->m_bBox.includes(atom.m_vPos)) // atom must be inside the bounding box
            {
                __debugbreak();
            }
        }

    }

    MyUnits<T> getFilteredAverageKin() const
    {
        return MyUnits<T>(m_averageKinFilter.getAverage());
    }

    void makeTimeStep()
    {
        updateListOfForces();

        this->propagate();

        MyUnits<T> fInstantaneousAverageKin = this->getInstantaneousAverageKin();
        m_averageKinFilter.addValue(fInstantaneousAverageKin.m_value);
        MyUnits<T> fFilteredAverageKin = getFilteredAverageKin();

        m_speedScaler.scale(fInstantaneousAverageKin, fFilteredAverageKin, this->m_atoms);
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
            auto& atom2 = this->m_atoms[uPoint2];
            for (NvU32 uTreePoint1 = (leafIndex == nodeIndex) ? uTreePoint2 + 1 : leafNode1.getFirstTreePoint(); uTreePoint1 < leafNode1.getEndTreePoint(); ++uTreePoint1)
            {
#if ASSERT_ONLY_CODE
                m_dbgNContributions += 2;
#endif
                NvU32 uPoint1 = m_ocTree.getPointIndex(uTreePoint1);
                auto& atom1 = this->m_atoms[uPoint1];
                auto vDir = this->m_bBox.computeDir(atom1, atom2);
                auto fLengthSqr = lengthSquared(vDir);
                if (fLengthSqr >= BondsDataBase<T>::s_zeroForceDistSqr) // if atoms are too far away - disregard
                {
                    continue;
                }

                for (NvU32 uBond1 = 0; ; ++uBond1)
                {
                    if (uBond1 >= atom1.getNBonds())
                    {
                        this->m_forces[ForceKey(uPoint1, uPoint2)]; // this adds default force between those two atoms into ForceMap
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

private:
    void updateListOfForces()
    {
        this->dissociateWeakBonds();

        m_ocTree.rebuild(removeUnits(this->m_bBox), (NvU32)this->m_atoms.size());

#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#endif

        m_ocTree.m_nodes[0].addNode2NodeInteractions(0, removeUnits(this->m_bBox), *this);
        nvAssert(m_dbgNContributions == this->m_atoms.size() * (this->m_atoms.size() - 1));
    }

    OcTree<Water> m_ocTree;
    RNGSobol m_rng;

    MyFilter<7> m_averageKinFilter;
    SpeedScaler<T> m_speedScaler;

#if ASSERT_ONLY_CODE
    NvU64 m_dbgNContributions = 0;
#endif
};