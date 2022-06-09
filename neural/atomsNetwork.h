#pragma once

#include "network.h"

struct ConstantAtomData
{
    float fMass = 0;
    float fValence = 0;
    float fElectroNegativity = 0;
};
struct TransientAtomData
{
    float3 vPos = {};
    float3 vSpeed = {};
    float fCharge = 0;
};
template <NvU32 nAtomsPerCluster>
struct ForceValues
{
    NvU32 nValues = 0;
    float fValues[nAtomsPerCluster];
};
template <NvU32 nAtomsPerCluster>
struct ForceIndices
{
    NvU32 atomIndices[nAtomsPerCluster];
};

template <class T, NvU32 nAtomsPerCluster>
struct AtomsNetwork : public NeuralNetwork<T>
{
    virtual bool startBatch_impl(std::vector<Tensor<T>*>& pInputs) override
    {
        return true;
    }
    virtual bool continueBatch_impl(std::vector<Tensor<T>*>& pInputs) override
    {
        return false;
    }
    virtual bool computeDeltaOutput_impl(std::vector<Tensor<T>*>& pOutputs) override
    {
        return true;
    }

    void init(std::vector<Atom<T>>& atoms)
    {
        copyStateToGPU(atoms, m_constAtomData);
    }
    void notifyStepBeginning(std::vector<Atom<T>>& atoms, ForceMap<T>& forceMap)
    {
        if (hasEnoughData())
            return;
        m_bStepStarted = true;

        if (m_pTransientAtomData.size() == 0)
        {
            m_pTransientAtomData.push_back(new GPUBuffer<TransientAtomData>);
            copyStateToGPU(atoms, **m_pTransientAtomData.rbegin());
            m_totalBytes += (*m_pTransientAtomData.rbegin())->getNBytes();
        }
        m_pForceValues.push_back(new GPUBuffer<ForceValues<nAtomsPerCluster>>);
        m_pForceIndices.push_back(new GPUBuffer<ForceIndices<nAtomsPerCluster>>);
        copyStateToGPU<true>(forceMap, **m_pForceValues.rbegin(), *m_pForceIndices.rbegin());
        m_totalBytes += (*m_pForceValues.rbegin())->getNBytes();
        m_totalBytes += (*m_pForceIndices.rbegin())->getNBytes();
    }
    size_t getNBytes() const { return m_totalBytes; }
    NvU32 getMaxClusterSize() const { return m_maxClusterSize; }
    void notifyStepDone(const std::vector<Atom<T>>& atoms, const ForceMap<T>& forceMap)
    {
        if (!m_bStepStarted)
            return;

        m_pTransientAtomData.push_back(new GPUBuffer<TransientAtomData>);
        m_totalBytes += (*m_pTransientAtomData.rbegin())->getNBytes();
        copyStateToGPU(atoms, **m_pTransientAtomData.rbegin());
        m_pForceValues.push_back(new GPUBuffer<ForceValues<nAtomsPerCluster>>);
        copyStateToGPU<false>(forceMap, **m_pForceValues.rbegin(), nullptr);

        m_totalBytes += (*m_pForceValues.rbegin())->getNBytes();
        m_bStepStarted = false;
    }
    bool hasEnoughData() const { return !m_bStepStarted && getNBytes() > 1024 * 1024 * 1; }
    void saveWeights();
    void loadWeights();
    void makePrediction();

private:
    void copyStateToGPU(const std::vector<Atom<T>>& atoms, GPUBuffer<ConstantAtomData> &dstBuffer)
    {
        std::vector<ConstantAtomData> &dst = dstBuffer.beginChanging();
        dst.resize(atoms.size());
        for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
        {
            const Atom<T>& srcAtom = atoms[uAtom];
            ConstantAtomData &dstAtom = dst[uAtom];
            dstAtom.fMass = (float)srcAtom.getMass();
            dstAtom.fValence = (float)srcAtom.getValence();
            dstAtom.fElectroNegativity = 1;
        }
        dstBuffer.endChanging();
    }
    void copyStateToGPU(const std::vector<Atom<T>>& atoms, GPUBuffer<TransientAtomData>& dstBuffer)
    {
        std::vector<TransientAtomData>& dst = dstBuffer.beginChanging();
        dst.resize(atoms.size());
        for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
        {
            const Atom<T>& srcAtom = atoms[uAtom];
            TransientAtomData& dstAtom = dst[uAtom];
            copy(dstAtom.vPos, srcAtom.m_vPos);
            copy(dstAtom.vSpeed, srcAtom.m_vSpeed);
            dstAtom.fCharge = 0;
        }
        dstBuffer.endChanging();
    }
    template <bool usePrevState>
    void copyStateToGPU(const ForceMap<T>& forceMap, GPUBuffer<ForceValues<nAtomsPerCluster>> &valuesBuffer, GPUBuffer<ForceIndices<nAtomsPerCluster>> *pIndicesBuffer)
    {
        std::vector<ForceValues<nAtomsPerCluster>>& pValues = valuesBuffer.beginChanging();
        std::vector<ForceIndices<nAtomsPerCluster>>* pIndices = pIndicesBuffer ? &pIndicesBuffer->beginChanging() : nullptr;
        pValues.resize(m_constAtomData.get().size());
        if (pIndices)
        {
            pIndices->resize(pValues.size());
        }
        for (NvU32 uForce = 0; uForce < forceMap.size(); ++uForce)
        {
            if (!forceMap.isValid(uForce))
                continue;
            const Force<T>& force = forceMap.accessForceByIndex(uForce);
            ForceValues<nAtomsPerCluster> &d1 = pValues[force.getAtom1Index()];
            ForceValues<nAtomsPerCluster> &d2 = pValues[force.getAtom2Index()];
            if (d1.nValues >= nAtomsPerCluster || d2.nValues >= nAtomsPerCluster)
            {
                __debugbreak(); // this means we have to increase nAtomsPerCluster
            }
            if (usePrevState)
            {
                d1.fValues[d1.nValues] = force.getPrevCovalentState() ? 1.f : 0.f;
                d2.fValues[d2.nValues] = force.getPrevCovalentState() ? 1.f : 0.f;
            }
            else
            {
                d1.fValues[d1.nValues] = force.isCovalentBond() ? 1.f : 0.f;
                d2.fValues[d2.nValues] = force.isCovalentBond() ? 1.f : 0.f;
            }
            if (pIndices)
            {
                (*pIndices)[force.getAtom1Index()].atomIndices[d1.nValues] = force.getAtom2Index();
                (*pIndices)[force.getAtom2Index()].atomIndices[d2.nValues] = force.getAtom1Index();
            }
            ++d1.nValues;
            ++d2.nValues;
            m_maxClusterSize = std::max(m_maxClusterSize, d1.nValues);
            m_maxClusterSize = std::max(m_maxClusterSize, d2.nValues);
        }
        valuesBuffer.endChanging();
        if (pIndicesBuffer)
        {
            pIndicesBuffer->endChanging();
        }
    }

    size_t m_totalBytes = 0;
    NvU32 m_maxClusterSize = 0;
    bool m_bStepStarted = false;

    GPUBuffer<ConstantAtomData> m_constAtomData;
    std::vector<GPUBuffer<TransientAtomData>*> m_pTransientAtomData;
    std::vector<GPUBuffer<ForceValues<nAtomsPerCluster>>*> m_pForceValues;
    std::vector<GPUBuffer<ForceIndices<nAtomsPerCluster>>*> m_pForceIndices;

    // those tensors are created from arrays above
    Tensor<float> m_input;
    Tensor<float> m_wantedOutput;
};
