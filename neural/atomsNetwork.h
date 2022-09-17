#pragma once

#include <memory>
#include <filesystem>
#include "../basics/vectors.h"
#include "../basics/simContext.h"
#include "network.h"
#include "batchTrainer.h"

struct ConstantAtomData
{
    float fMass = 0;
    float fValence = 0;
    float fElectroNegativity = 0;
};
struct TransientAtomData
{
    kirill::float3 vPos = {};
    kirill::float3 vSpeed = {};
    float fCharge = 0;
};
// ForceValues and ForceIndices are split to different structs because ForceIndices at the beginning and
// at the end of each step are the same, while ForceValues change
template <NvU32 nAtomsPerCluster>
struct ForceValues
{
    std::array<float, nAtomsPerCluster - 1> m_nCovalentBonds = {};
};
template <NvU32 nAtomsPerCluster>
struct ForceIndices
{
    NvU32 nIndices = 0;
    std::array<NvU32, nAtomsPerCluster - 1> atomIndices = {};
};


template <class T>
inline void copy(rtvector<float, 3>& dst, const rtvector<T, 3>& src)
{
    dst[0] = (float)src[0];
    dst[1] = (float)src[1];
    dst[2] = (float)src[2];
}

template <class T, NvU32 nAtomsPerCluster>
struct AtomsNetwork : public NeuralNetwork
{
    AtomsNetwork()
    {
        createaAtomLayers(m_pLayers);
    }
    using InternalLayerType = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;

    inline NvU32 getNAtoms() const { return (NvU32)m_constAtomData.size(); }

    void init(const SimContext<T>& simContext)
    {
        copyConstAtomsDataFromTheModel(simContext.m_atoms);
        m_boxWrapper = simContext.m_bBox;
    }

    void propagate(SimContext<T>& simContext)
    {
        m_pTransientAtomData.resize(0);
        copyTransientAtomsDataFromTheModel(simContext.m_atoms);
        m_pForceIndices.resize(0);
        copyForceIndicesFromTheModel(simContext);
        m_pForceValues.resize(0);
        copyBondsFromTheModel(simContext.m_forces);

        NvU32 nAtoms = (NvU32)simContext.m_atoms.size();
        if (m_batchAtomIndices.size() != nAtoms)
        {
            m_batchAtomIndices.resize(simContext.m_atoms.size());
            for (NvU32 u = 0; u < m_batchAtomIndices.size(); ++u)
            {
                m_batchAtomIndices[u] = u;
            }
        }

        TensorRef pInput = createInputTensor(m_batchAtomIndices, 0, (NvU32)m_batchAtomIndices.size());
        TensorRef pOutput = this->forwardPass(0, pInput);
        pOutput->syncToHost();

        Tensor<float>& output = *pOutput;
        for (NvU32 u = 0; u < nAtoms; ++u)
        {
            propagateAtom(output, u, simContext);
        }
    }

    void notifyStepBeginning(const SimContext<T> &simContext)
    {
        if (hasEnoughData())
        {
            if (m_bNeedToSave)
            {
                m_bNeedToSave = false;
                MyWriter writer("c:\\atomNets\\networkFromWaterApp.bin");
                serialize(writer);
            }
            return;
        }
        m_bSimStepStarted = true;
        m_bNeedToSave = true;

        if (m_pTransientAtomData.size() == 0)
        {
            copyTransientAtomsDataFromTheModel(simContext.m_atoms);
        }
        copyForceIndicesFromTheModel(simContext);
        copyBondsFromTheModel(simContext.m_forces);
    }

    void notifyStepDone(const SimContext<T> &simContext)
    {
        if (!m_bSimStepStarted)
            return;

        copyTransientAtomsDataFromTheModel(simContext.m_atoms);
        copyBondsFromTheModel(simContext.m_forces);

        m_bSimStepStarted = false;
    }
    NvU32 getNStoredSimSteps() const { return (NvU32)m_pForceIndices.size(); }
    bool hasEnoughData() const { return !m_bSimStepStarted && getNStoredSimSteps() >= 1000; }

    virtual NvU32 getNBatches()
    {
        return getNStoredSimSteps() * getNAtoms() / NATOMS_PER_BATCH;
    }

    TensorRef createInputTensor(const std::vector<NvU32>& atomIndices, NvU32 uFirstAtom,
        NvU32 nAtoms) const
    {
        // initialize inputs and outputs to zero (force CPU because we have to fill those buffers on CPU)
        std::array<unsigned, 4> inputDims = { nAtoms, s_nInputValuesPerCluster, 1, 1 };
        TensorRef pInput = std::make_shared<Tensor<float>>(inputDims);
        Tensor<float>& input = *pInput;
        input.clearSubregion(0, (NvU32)input.size(), EXECUTE_MODE_FORCE_CPU);

        // copy all data to input and output tensors
        for (NvU32 u = 0; u < nAtoms; ++u)
        {
            copyClusterToInputTensor(input, u, atomIndices[uFirstAtom + u]);
        }

        return pInput;
    }
    TensorRef allocateOutputTensor(NvU32 nAtoms) const
    {
        std::array<unsigned, 4> outputDims = { nAtoms, s_nOutputValuesPerCluster, 1, 1 };
        TensorRef pOutput = std::make_shared<Tensor<float>>(outputDims);
        pOutput->init(outputDims);
        pOutput->clearSubregion(0, (NvU32)pOutput->size(), EXECUTE_MODE_FORCE_CPU);
        return pOutput;
    }

    virtual void initBatch(BatchTrainer &batchTrainer, NvU32 uBatch) override
    {
        if (m_batchAtomIndices.size() == 0)
        {
            NvU32 nTotalClusters = getNBatches() * NATOMS_PER_BATCH;
            createBatchAtomIndices(nTotalClusters);
        }
        TensorRef pInput = createInputTensor(m_batchAtomIndices, uBatch * NATOMS_PER_BATCH, NATOMS_PER_BATCH);

        TensorRef pWantedOutput = allocateOutputTensor(NATOMS_PER_BATCH);
        Tensor<float>& wantedOutput = *pWantedOutput;
        // copy data to output tensors
        for (NvU32 u = 0; u < NATOMS_PER_BATCH; ++u)
        {
            copyClusterToOutputTensor(wantedOutput, u,
                m_batchAtomIndices[uBatch * NATOMS_PER_BATCH + u]);
        }
        batchTrainer.init(*this, uBatch, pInput, pWantedOutput);
    }

    virtual void serialize(ISerializer& s) override
    {
        NeuralNetwork::serialize(s);
        m_constAtomData.serialize("m_constAtomDataTensor", s);
        s.serializeArrayOfPointers("m_pTransientAtomDataArray", m_pTransientAtomData);
        s.serializeArrayOfPointers("m_pForceValuesArray", m_pForceValues);
        s.serializeArrayOfPointers("m_pForceIndicesArray", m_pForceIndices);
        s.serializePreallocatedMem("m_boxWrapper", &m_boxWrapper, sizeof(m_boxWrapper));
    }

private:
    // **** offsets we use to create input tensor
    static constexpr NvU32 computeInputAtomOffset(NvU32 uAtom)
    {
        NvU32 inputAtomSizeInBytes = sizeof(ConstantAtomData) + sizeof(TransientAtomData);
        nvAssert(inputAtomSizeInBytes % sizeof(float) == 0);
        return uAtom * (inputAtomSizeInBytes / sizeof(float));
    }
    static constexpr NvU32 computeInputForceOffset(NvU32 uForce)
    {
        // each force is a single float (stores only the number of covalent bonds)
        // we'll put central atom at the end, and that atom doesn't need transient data (because it's all zero)
        nvAssert(sizeof(TransientAtomData) % sizeof(float) == 0);
        return computeInputAtomOffset(nAtomsPerCluster) + uForce - sizeof(TransientAtomData) / sizeof(float);
    }
    // this is essentially an offset to the force one after the last
    static const NvU32 s_nInputValuesPerCluster = computeInputForceOffset(nAtomsPerCluster);

    // **** output offsets computation
    static constexpr NvU32 computeOutputAtomOffset(NvU32 uAtom)
    {
        nvAssert(sizeof(TransientAtomData) % sizeof(float) == 0);
        return uAtom * (sizeof(TransientAtomData) / sizeof(float));
    }
    static constexpr NvU32 computeOutputForceOffset(NvU32 uForce)
    {
        // we output just one atom for now, and each force stores just single float (number of covalent bonds)
        return computeOutputAtomOffset(1) + uForce;
    }
    // this is essentially an offset to the force one after the last
    static const NvU32 s_nOutputValuesPerCluster = computeOutputForceOffset(nAtomsPerCluster);

    bool createaAtomLayers(std::vector<std::shared_ptr<ILayer>>& pLayers)
    {
        nvAssert(pLayers.size() == 0);

        std::array<std::array<unsigned, 4>, 3> outputDims =
        { {
            { 1, 128, 1, 1 },
            { 1, 128, 1, 1 },
            { 1, 128, 1, 1 },
        } };
        std::array<unsigned, 4> prevOutputDims = { 1, s_nInputValuesPerCluster, 1, 1 };
        for (NvU32 u = 0; u < outputDims.size(); ++u)
        {
            auto pLayer = std::make_shared<InternalLayerType>();
            pLayer->init(prevOutputDims, outputDims[u]);
            pLayers.push_back(pLayer);
            prevOutputDims = outputDims[u];
        }
        using OutputLayerType = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        auto pLayer = std::make_shared<OutputLayerType>();
        std::array<unsigned, 4> wantedOutputDims = { 1, s_nOutputValuesPerCluster, 1, 1 };
        pLayer->init(prevOutputDims, wantedOutputDims);
        pLayers.push_back(pLayer);
        return true;
    }
    static const NvU32 NATOMS_PER_BATCH = 256; // number of atoms we train on simultaneously

    void copyClusterToInputTensor(Tensor<float> &input, NvU32 uDstCluster,
        NvU32 uSrcCluster) const
    {
        NvU32 uSimStep = uSrcCluster / getNAtoms();
        NvU32 uCentralAtom = uSrcCluster % getNAtoms();

        const GPUBuffer<TransientAtomData>& transientData = *m_pTransientAtomData[uSimStep];
        const ForceIndices<nAtomsPerCluster>& fi = (*m_pForceIndices[uSimStep])[uCentralAtom];
        const ForceValues<nAtomsPerCluster>& fv = (*m_pForceValues[uSimStep * 2])[uCentralAtom];

        copyAtomToInputTensor(input, uDstCluster, nAtomsPerCluster - 1, uCentralAtom, uCentralAtom, transientData); // copy the central atom
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            copyAtomToInputTensor(input, uDstCluster, u, uCentralAtom, fi.atomIndices[u], transientData); // copy auxiliary atoms
            input.access(uDstCluster, computeInputForceOffset(u), 0, 0) = fv.m_nCovalentBonds[u]; // copy the force information
        }
    }
    void copyAtomToInputTensor(Tensor<float>& input, NvU32 uDstCluster, NvU32 uDstSlot,
        NvU32 uCentralAtom, NvU32 uSrcAtom,
        const GPUBuffer<TransientAtomData> &transientData) const
    {
        NvU32 hi = computeInputAtomOffset(uDstSlot);
        const ConstantAtomData& constData = m_constAtomData[uSrcAtom];
        const TransientAtomData& transData = transientData[uSrcAtom];
        const TransientAtomData& centData = transientData[uCentralAtom];
        input.access(uDstCluster, hi++, 0, 0) = constData.fElectroNegativity;
        input.access(uDstCluster, hi++, 0, 0) = constData.fMass;
        input.access(uDstCluster, hi++, 0, 0) = constData.fValence;
        input.access(uDstCluster, hi++, 0, 0) = transData.fCharge;
        // store the vector from the central atom to this atom
        rtvector<float, 3> vPos = m_boxWrapper.computeDir(transData.vPos, centData.vPos);
        input.access(uDstCluster, hi++, 0, 0) = vPos[0];
        input.access(uDstCluster, hi++, 0, 0) = vPos[1];
        input.access(uDstCluster, hi++, 0, 0) = vPos[2];
        rtvector<float, 3> vSpeed = transData.vSpeed - centData.vSpeed;
        input.access(uDstCluster, hi++, 0, 0) = vSpeed[0];
        input.access(uDstCluster, hi++, 0, 0) = vSpeed[1];
        input.access(uDstCluster, hi++, 0, 0) = vSpeed[2];
    }
    void copyClusterToOutputTensor(Tensor<float> &wantedOutput, NvU32 uDstCluster, NvU32 uSrcCluster)
    {
        NvU32 uSimStep = uSrcCluster / getNAtoms();
        NvU32 uCentralAtom = uSrcCluster % getNAtoms();

        const GPUBuffer<TransientAtomData>& transientDataPrev = *m_pTransientAtomData[uSimStep];
        const GPUBuffer<TransientAtomData>& transientDataNext = *m_pTransientAtomData[uSimStep + 1];
        const TransientAtomData& centAtomIn = transientDataPrev[uCentralAtom];
        const TransientAtomData& centAtomOut = transientDataNext[uCentralAtom];
        const ForceIndices<nAtomsPerCluster>& fi = (*m_pForceIndices[uSimStep])[uCentralAtom];
        const ForceValues<nAtomsPerCluster>& fv = (*m_pForceValues[uSimStep * 2 + 1])[uCentralAtom];

        // copy the central atom
        rtvector<float, 3> vPos = m_boxWrapper.computeDir(centAtomOut.vPos, centAtomIn.vPos);
        wantedOutput.access(uDstCluster, 0, 0, 0) = vPos[0];
        wantedOutput.access(uDstCluster, 1, 0, 0) = vPos[1];
        wantedOutput.access(uDstCluster, 2, 0, 0) = vPos[2];
        rtvector<float, 3> vSpeed = centAtomOut.vSpeed - centAtomIn.vSpeed;
        wantedOutput.access(uDstCluster, 3, 0, 0) = vSpeed[0];
        wantedOutput.access(uDstCluster, 4, 0, 0) = vSpeed[1];
        wantedOutput.access(uDstCluster, 5, 0, 0) = vSpeed[2];

        // copy the force information
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            nvAssert(computeOutputForceOffset(u) == 6 + u);
            wantedOutput.access(uDstCluster, 6 + u, 0, 0) = fv.m_nCovalentBonds[u];
        }
    }

    void propagateAtom(Tensor<float>& output, NvU32 uAtom, SimContext<T>& simContext)
    {
        Atom<T>& atom = simContext.m_atoms[uAtom];
        rtvector<float, 3> vDir;
        vDir[0] = output.access(uAtom, 0, 0, 0);
        vDir[1] = output.access(uAtom, 1, 0, 0);
        vDir[2] = output.access(uAtom, 2, 0, 0);
        atom.m_vPos = m_boxWrapper.wrapThePos(atom.m_vPos + vDir);
        rtvector<float, 3> vSpeed;
        vSpeed[0] = output.access(uAtom, 3, 0, 0);
        vSpeed[1] = output.access(uAtom, 4, 0, 0);
        vSpeed[2] = output.access(uAtom, 5, 0, 0);
        atom.m_vSpeed += vSpeed;

        const ForceIndices<nAtomsPerCluster>& fi = (*m_pForceIndices[0])[uAtom];
        const ForceValues<nAtomsPerCluster>& fv = (*m_pForceValues[0])[uAtom];
        // copy the force information
        bool bDeferredCovalentBonds = false;
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            nvAssert(computeOutputForceOffset(u) == 6 + u);
            float nCovalentBonds = round(output.access(uAtom, 6 + u, 0, 0));
            if (nCovalentBonds > fv.m_nCovalentBonds[u]) // new covalent bonds have appeared
            {
                NvU32 uAtom1 = fi.atomIndices[u];
                NvU32 uForce = simContext.m_forces.findExistingForce(uAtom, uAtom1);
                Force<T>& force = simContext.m_forces.accessForceByIndex(uForce);
                Atom<T>& atom1 = simContext.m_atoms[uAtom1];
                auto vDir = m_boxWrapper.computeDir(atom.m_vPos, atom1.m_vPos);
                T fDistSqr = lengthSquared(vDir);
                force.createCovalentBondIfNeeded(atom, atom1, fDistSqr);
            }
        }
    }
    void copyConstAtomsDataFromTheModel(const std::vector<Atom<T>>& atoms)
    {
        GPUBuffer<ConstantAtomData>& dst = m_constAtomData;
        dst.resize(atoms.size());

        for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
        {
            const Atom<T>& srcAtom = atoms[uAtom];
            ConstantAtomData &dstAtom = dst[uAtom];
            dstAtom.fMass = (float)srcAtom.getMass();
            dstAtom.fValence = (float)srcAtom.getValence();
            dstAtom.fElectroNegativity = 1;
        }
    }
    void copyTransientAtomsDataFromTheModel(const std::vector<Atom<T>>& atoms)
    {
        m_pTransientAtomData.push_back(new GPUBuffer<TransientAtomData>);
        GPUBuffer<TransientAtomData>& dst = **m_pTransientAtomData.rbegin();
        dst.resize(atoms.size());

        for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
        {
            const Atom<T>& srcAtom = atoms[uAtom];
            TransientAtomData& dstAtom = dst[uAtom];
            copy(dstAtom.vPos, srcAtom.m_vPos);
            copy(dstAtom.vSpeed, srcAtom.m_vSpeed);
            dstAtom.fCharge = 0;
        }
    }
    NvU32 findTheFarthestAtom(NvU32 uAtom, const ForceIndices<nAtomsPerCluster>& indices, const SimContext<T> &simContext)
    {
        const std::vector<Atom<T>>& atoms = simContext.m_atoms;
        const BoxWrapper<T>& boxWrapper = simContext.m_bBox;
        const Atom<T>& atom1 = atoms[uAtom];
        NvU32 uFarthestIndex = 0xffffffff;
        T fMaxDistSqr = 0;
        for (NvU32 u = 0; u < indices.nIndices; ++u)
        {
            const Atom<T>& atom2 = atoms[indices.atomIndices[u]];
            rtvector<MyUnits<T>, 3> dir = boxWrapper.computeDir(atom1.m_vPos, atom2.m_vPos);
            T fDistSqr = dot(dir, dir);
            if (fDistSqr > fMaxDistSqr)
            {
                fMaxDistSqr = fDistSqr;
                uFarthestIndex = u;
            }
        }
        return uFarthestIndex;
    }
    void copyForceIndicesFromTheModel(const SimContext<T>& simContext)
    {
        const ForceMap<T>& forceMap = simContext.m_forces;

        m_pForceIndices.push_back(new GPUBuffer<ForceIndices<nAtomsPerCluster>>);
        GPUBuffer<ForceIndices<nAtomsPerCluster>>& pIndices = **m_pForceIndices.rbegin();
        pIndices.resize(getNAtoms());

        for (NvU32 uForce = 0; uForce < forceMap.size(); ++uForce)
        {
            if (!forceMap.isValid(uForce))
                continue;
            const Force<T>& force = forceMap.accessForceByIndex(uForce);
            {
                ForceIndices<nAtomsPerCluster>& fi1 = pIndices[force.getAtom1Index()];
                if (fi1.nIndices >= fi1.atomIndices.size())
                {
                    --fi1.nIndices;
                    // find the farthest atom and place it at the last position (it will be replaced)
                    NvU32 iFarthest = findTheFarthestAtom(force.getAtom1Index(), fi1, simContext);
                    nvSwap(fi1.atomIndices[iFarthest], fi1.atomIndices[fi1.nIndices]);
                }
                fi1.atomIndices[fi1.nIndices++] = force.getAtom2Index();
            }
            {
                ForceIndices<nAtomsPerCluster>& fi2 = pIndices[force.getAtom2Index()];
                if (fi2.nIndices >= fi2.atomIndices.size())
                {
                    --fi2.nIndices;
                    // find the farthest atom and place it at the last position (it will be replaced)
                    NvU32 iFarthest = findTheFarthestAtom(force.getAtom2Index(), fi2, simContext);
                    nvSwap(fi2.atomIndices[iFarthest], fi2.atomIndices[fi2.nIndices]);
                }
                fi2.atomIndices[fi2.nIndices++] = force.getAtom1Index();
            }
        }
    }
    void copyBondsFromTheModel(const ForceMap<T>& forceMap)
    {
        m_pForceValues.push_back(new GPUBuffer<ForceValues<nAtomsPerCluster>>);
        GPUBuffer<ForceValues<nAtomsPerCluster>>& pValues = **m_pForceValues.rbegin();
        pValues.resize(getNAtoms());

        const GPUBuffer<ForceIndices<nAtomsPerCluster>>& pIndices = **m_pForceIndices.rbegin();
        for (NvU32 uAtom1 = 0; uAtom1 < getNAtoms(); ++uAtom1)
        {
            const ForceIndices<nAtomsPerCluster>& indices = pIndices[uAtom1];
            for (NvU32 u = 0; u < indices.nIndices; ++u)
            {
                NvU32 uAtom2 = indices.atomIndices[u];
                NvU32 uForce = forceMap.findExistingForce(uAtom1, uAtom2);
                const Force<T>& force = forceMap.accessForceByIndex(uForce);
                if (force.isCovalentBond())
                {
                    pValues[uAtom1].m_nCovalentBonds[u] = 1;
                }
            }
        }
    }

    bool m_bSimStepStarted = false, m_bNeedToSave = false;

    GPUBuffer<ConstantAtomData> m_constAtomData; // 1 buffer - describes static properties of all simulated atoms
    std::vector<GPUBuffer<TransientAtomData>*> m_pTransientAtomData; // 1 in the beginning + 1 per simulation step
    std::vector<GPUBuffer<ForceValues<nAtomsPerCluster>>*> m_pForceValues; // 2 per simulation step
    std::vector<GPUBuffer<ForceIndices<nAtomsPerCluster>>*> m_pForceIndices; // 1 per simulation step
    BoxWrapper<T> m_boxWrapper;

    std::vector<NvU32> m_batchAtomIndices;
    void createBatchAtomIndices(NvU32 nIndices)
    {
        m_batchAtomIndices.resize(nIndices);
        for (NvU32 u = 0; u < nIndices; ++u)
        {
            m_batchAtomIndices[u] = u;
        }
        RNGUniform rng;
        // scramble the array
        for (NvU32 u = 0; u < m_batchAtomIndices.size(); ++u)
        {
            NvU32 u1 = rng.generateUnsigned(0, (NvU32)m_batchAtomIndices.size());
            nvSwap(m_batchAtomIndices[u1], m_batchAtomIndices[u]);
        }
    }

    RNGUniform m_rng;
};
