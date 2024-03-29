#pragma once

#include <memory>
#include <filesystem>
#include "../basics/vectors.h"
#include "../basics/simContext.h"
#include "network.h"
#include "batch.h"

// classes used to create scrambled arrays of atom indices for training
struct NullScrambler
{
    NullScrambler(NvU32 uStart, NvU32 uSize) : m_uStart(uStart), m_uSize(uSize)
    {
    }
    NvU32 operator[](NvU32 u) const
    {
        nvAssert(u < m_uSize);
        return m_uStart + u;
    }
    NvU32 size() const { return m_uSize; }
private:
    NvU32 m_uStart = 0, m_uSize;
};
struct ArrayScrambler
{
    ArrayScrambler(const std::vector<NvU32>& p, NvU32 uStart, NvU32 uSize) : m_p (&p[uStart]), m_uSize(uSize)
    {
        nvAssert(uStart + uSize <= p.size());
    }
    NvU32 operator[](NvU32 u) const
    {
        nvAssert(u < m_uSize);
        return m_p[u];
    }
    NvU32 size() const { return m_uSize; }
private:
    const NvU32* m_p = nullptr;
    NvU32 m_uSize = 0;
};

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

constexpr NvU32 ATOMS_PER_CLUSTER = 64;

// ForceValues and ForceIndices are split to different structs because ForceIndices at the beginning and
// at the end of each step are the same, while ForceValues change
struct ForceValues
{
    std::array<float, ATOMS_PER_CLUSTER - 1> m_nCovalentBonds = {};
};
struct ForceIndices
{
    NvU32 nIndices = 0;
    std::array<NvU32, ATOMS_PER_CLUSTER - 1> atomIndices = {};
};


template <class T>
inline void copy(rtvector<float, 3>& dst, const rtvector<T, 3>& src)
{
    dst[0] = (float)src[0];
    dst[1] = (float)src[1];
    dst[2] = (float)src[2];
}

template <class T>
struct AtomsDataLoader : public DataLoader
{
    using InternalLayerType = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;

    inline NvU32 getNAtoms() const { return (NvU32)m_constAtomData.size(); }

    void init(const SimContext<T>& simContext)
    {
        copyConstAtomsDataFromTheModel(simContext.m_atoms);
        m_boxWrapper = simContext.m_bBox;
    }

#if 0 // not needed for now - but good code when (for when I do inference)
    void computeNeuralDir(SimContext<T>& simContext)
    {
        NvU32 nAtoms = getNAtoms();
        NullScrambler scrambler(0, nAtoms);

        // the scrambler points to the first sim step - so put our last step there temporarily
        nvSwap(m_lastSimStep, m_simSteps[0]);
        TensorRef pInput = createInputTensor(scrambler);
        nvSwap(m_lastSimStep, m_simSteps[0]);

        TensorRef pOutput = this->forwardPass(0, pInput);
        pOutput->syncToHost();

        Tensor& output = *pOutput;
        for (NvU32 u = 0; u < nAtoms; ++u)
        {
            computeNeuralDir(output, u, simContext);
        }
    }
#endif

    void notifyStepBeginning(const SimContext<T> &simContext, NvU32 stepIndex)
    {
        // do we have atoms from the previous sim step that we can use?
        if (m_lastSimStep.m_pAtomsNext && stepIndex == m_lastSimStep.m_index + 1)
            m_lastSimStep.m_pAtomsPrev = m_lastSimStep.m_pAtomsNext;
        else
            m_lastSimStep.m_pAtomsPrev = copyTransientAtomsDataFromTheModel(simContext.m_atoms);
        m_lastSimStep.m_pAtomsNext = nullptr;
        m_lastSimStep.m_index = stepIndex;
        m_lastSimStep.m_pForceIndices = copyForceIndicesFromTheModel(simContext);
        m_lastSimStep.m_pForcesPrev = copyBondsFromTheModel(simContext.m_forces, *m_lastSimStep.m_pForceIndices);
        m_lastSimStep.m_pForcesNext = nullptr;
    }

    void notifyStepDone(const SimContext<T> &simContext)
    {
        m_lastSimStep.m_pAtomsNext = copyTransientAtomsDataFromTheModel(simContext.m_atoms);
        m_lastSimStep.m_pForcesNext = copyBondsFromTheModel(simContext.m_forces, *m_lastSimStep.m_pForceIndices);

        static const NvU32 NMAX_SIM_STEPS = 10000;

        if (m_simSteps.size() <= NMAX_SIM_STEPS)
        {
            if (m_simSteps.size() < NMAX_SIM_STEPS)
            {
                m_simSteps.push_back(m_lastSimStep);
            }
            if (m_simSteps.size() == NMAX_SIM_STEPS)
            {
                char sBuffer[128];
                sprintf_s(sBuffer, "c:\\atomNets\\networkFromWaterApp_%d.bin", (NvU32)m_simSteps.size());
                MyWriter writer(sBuffer);
                serialize(writer);
            }
        }
    }
    NvU32 getNStoredSimSteps() const { return (NvU32)m_simSteps.size(); }

    virtual NvU32 getNBatches()
    {
        return getNStoredSimSteps() * getNAtoms() / NATOMS_PER_BATCH;
    }

    virtual void serialize(ISerializer& s)
    {
        DataLoader::serialize(s);
        m_constAtomData.serialize("m_constAtomDataTensor", s);
        s.serializeArraySize("m_simSteps", m_simSteps);
        if (m_simSteps.size() > 0)
        {
            m_simSteps[0].serialize(s);
        }
        for (NvU32 u = 1; u < m_simSteps.size(); ++u)
        {
            m_simSteps[u].serialize(s, &m_simSteps[u - 1]);
        }
        s.serializePreallocatedMem("m_boxWrapper", &m_boxWrapper, sizeof(m_boxWrapper));
    }

private:
    template <class Scrambler>
    std::shared_ptr<Batch> initBatchInternal(NvU32 uBatch, const Scrambler &scrambler)
    {
        TensorRef pInput = createInputTensor(scrambler);

        TensorRef pWantedOutput = allocateOutputTensor(scrambler.size());
        Tensor& wantedOutput = *pWantedOutput;
        // copy data to output tensors
        for (NvU32 u = 0; u < scrambler.size(); ++u)
        {
            copyClusterToOutputTensor(wantedOutput, u, scrambler[u]);
        }
        return std::make_shared<Batch>(uBatch, pInput, pWantedOutput);
    }
    virtual std::shared_ptr<Batch> createBatch(NvU32 uBatch) override
    {
        if (m_batchAtomIndices.size() == 0)
        {
            NvU32 nTotalClusters = getNBatches() * NATOMS_PER_BATCH;
            createBatchAtomIndices(nTotalClusters);
        }
        ArrayScrambler scrambler(m_batchAtomIndices, uBatch * NATOMS_PER_BATCH, NATOMS_PER_BATCH);
        return initBatchInternal(uBatch, scrambler);
    }

    template <class Scrambler>
    TensorRef createInputTensor(const Scrambler& scrambler) const
    {
        // initialize inputs and outputs to zero (force CPU because we have to fill those buffers on CPU)
        std::array<unsigned, 4> inputDims = { scrambler.size(), s_nInputValuesPerCluster, 1, 1};
        TensorRef pInput = std::make_shared<Tensor>(inputDims, sizeof(float));
        Tensor& input = *pInput;
        input.clearSubregion(0, (NvU32)input.size(), EXECUTE_MODE_FORCE_CPU);

        // copy all data to input and output tensors
        for (NvU32 u = 0; u < scrambler.size(); ++u)
        {
            copyClusterToInputTensor(input, u, scrambler[u]);
        }

        return pInput;
    }

    TensorRef allocateOutputTensor(NvU32 nAtoms) const
    {
        std::array<unsigned, 4> outputDims = { nAtoms, s_nOutputValuesPerCluster, 1, 1 };
        TensorRef pOutput = std::make_shared<Tensor>(outputDims, sizeof(float));
        pOutput->clearSubregion(0, (NvU32)pOutput->size(), EXECUTE_MODE_FORCE_CPU);
        return pOutput;
    }

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
        return computeInputAtomOffset(ATOMS_PER_CLUSTER) + uForce - sizeof(TransientAtomData) / sizeof(float);
    }
    // this is essentially an offset to the force one after the last
    static const NvU32 s_nInputValuesPerCluster = computeInputForceOffset(ATOMS_PER_CLUSTER);

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
    static const NvU32 s_nOutputValuesPerCluster = computeOutputForceOffset(ATOMS_PER_CLUSTER);

public:
    std::shared_ptr<NeuralNetwork> createNetwork() override
    {
        std::shared_ptr<NeuralNetwork> pNetwork = std::make_shared<NeuralNetwork>();
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
            pNetwork->addLayer(pLayer);
            prevOutputDims = outputDims[u];
        }
        using OutputLayerType = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        auto pLayer = std::make_shared<OutputLayerType>();
        std::array<unsigned, 4> wantedOutputDims = { 1, s_nOutputValuesPerCluster, 1, 1 };
        pLayer->init(prevOutputDims, wantedOutputDims);
        pNetwork->addLayer(pLayer);
        return pNetwork;
    }

private:
    static const NvU32 NATOMS_PER_BATCH = 256; // number of atoms we train on simultaneously

    void copyClusterToInputTensor(Tensor &input, NvU32 uDstCluster,
        NvU32 uSrcCluster) const
    {
        NvU32 uSimStep = uSrcCluster / getNAtoms();
        NvU32 uCentralAtom = uSrcCluster % getNAtoms();

        const Step& simStep = m_simSteps[uSimStep];
        const GPUBuffer& transientData = *simStep.m_pAtomsPrev;
        const GPUBuffer& pForceIndices = (*simStep.m_pForceIndices);
        const ForceIndices& fi = pForceIndices.as<ForceIndices>(uCentralAtom);
        const GPUBuffer& pForceValues = (*simStep.m_pForcesPrev);
        const ForceValues& fv = pForceValues.as<ForceValues>(uCentralAtom);

        copyAtomToInputTensor(input, uDstCluster, ATOMS_PER_CLUSTER - 1, uCentralAtom, uCentralAtom, transientData); // copy the central atom
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            copyAtomToInputTensor(input, uDstCluster, u, uCentralAtom, fi.atomIndices[u], transientData); // copy auxiliary atoms
            input.access<float>(uDstCluster, computeInputForceOffset(u), 0, 0) = fv.m_nCovalentBonds[u]; // copy the force information
        }
    }
    void copyAtomToInputTensor(Tensor& input, NvU32 uDstCluster, NvU32 uDstSlot,
        NvU32 uCentralAtom, NvU32 uSrcAtom,
        const GPUBuffer &transientData) const
    {
        NvU32 hi = computeInputAtomOffset(uDstSlot);
        const ConstantAtomData& constData = m_constAtomData.as<ConstantAtomData>(uSrcAtom);
        const TransientAtomData& transData = transientData.as<TransientAtomData>(uSrcAtom);
        const TransientAtomData& centData = transientData.as<TransientAtomData>(uCentralAtom);
        input.access<float>(uDstCluster, hi++, 0, 0) = constData.fElectroNegativity;
        input.access<float>(uDstCluster, hi++, 0, 0) = constData.fMass;
        input.access<float>(uDstCluster, hi++, 0, 0) = constData.fValence;
        input.access<float>(uDstCluster, hi++, 0, 0) = transData.fCharge;
        // store the vector from the central atom to this atom
        rtvector<float, 3> vPos = m_boxWrapper.computeDir(transData.vPos, centData.vPos);
        input.access<float>(uDstCluster, hi++, 0, 0) = vPos[0];
        input.access<float>(uDstCluster, hi++, 0, 0) = vPos[1];
        input.access<float>(uDstCluster, hi++, 0, 0) = vPos[2];
        rtvector<float, 3> vSpeed = transData.vSpeed - centData.vSpeed;
        input.access<float>(uDstCluster, hi++, 0, 0) = vSpeed[0];
        input.access<float>(uDstCluster, hi++, 0, 0) = vSpeed[1];
        input.access<float>(uDstCluster, hi++, 0, 0) = vSpeed[2];
    }
    void copyClusterToOutputTensor(Tensor &wantedOutput, NvU32 uDstCluster,
        NvU32 uSrcCluster) const
    {
        NvU32 uSimStep = uSrcCluster / getNAtoms();
        NvU32 uCentralAtom = uSrcCluster % getNAtoms();

        const Step& simStep = m_simSteps[uSimStep];
        const GPUBuffer& transientDataPrev = *simStep.m_pAtomsPrev;
        const GPUBuffer& transientDataNext = *simStep.m_pAtomsNext;
        const TransientAtomData& centAtomIn = transientDataPrev.as<TransientAtomData>(uCentralAtom);
        const TransientAtomData& centAtomOut = transientDataNext.as<TransientAtomData>(uCentralAtom);
        const GPUBuffer& pForceIndices = *simStep.m_pForceIndices;
        const ForceIndices& fi = pForceIndices.as<ForceIndices>(uCentralAtom);
        const GPUBuffer& pForceValues = *simStep.m_pForcesNext;
        const ForceValues& fv = pForceValues.as<ForceValues>(uCentralAtom);

        // copy the central atom
        rtvector<float, 3> vPos = m_boxWrapper.computeDir(centAtomOut.vPos, centAtomIn.vPos);
        wantedOutput.access<float>(uDstCluster, 0, 0, 0) = vPos[0];
        wantedOutput.access<float>(uDstCluster, 1, 0, 0) = vPos[1];
        wantedOutput.access<float>(uDstCluster, 2, 0, 0) = vPos[2];
        rtvector<float, 3> vSpeed = centAtomOut.vSpeed - centAtomIn.vSpeed;
        wantedOutput.access<float>(uDstCluster, 3, 0, 0) = vSpeed[0];
        wantedOutput.access<float>(uDstCluster, 4, 0, 0) = vSpeed[1];
        wantedOutput.access<float>(uDstCluster, 5, 0, 0) = vSpeed[2];
        // charge is 0 currently - so we don't need to assign it here

        // copy the force information
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            nvAssert(computeOutputForceOffset(u) == 7 + u);
            wantedOutput.access<float>(uDstCluster, 7 + u, 0, 0) = fv.m_nCovalentBonds[u];
        }
    }

#if 0 // not needed for now but good code (for when I do inference)
    void computeNeuralDir(Tensor& output, NvU32 uAtom, SimContext<T>& simContext) const
    {
        Atom<T>& atom = simContext.m_atoms[uAtom];
        rtvector<float, 3> vDir;
        vDir[0] = output.access(uAtom, 0, 0, 0);
        vDir[1] = output.access(uAtom, 1, 0, 0);
        vDir[2] = output.access(uAtom, 2, 0, 0);
        atom.m_vNeuralDir = vDir;
#if 0
        rtvector<float, 3> vSpeed;
        vSpeed[0] = output.access(uAtom, 3, 0, 0);
        vSpeed[1] = output.access(uAtom, 4, 0, 0);
        vSpeed[2] = output.access(uAtom, 5, 0, 0);
        atom.m_vSpeed += vSpeed;

        const Step &simStep = *m_simSteps.rbegin();
        const ForceIndices& fi = (*simStep.m_pForceIndices)[uAtom];
        const ForceValues& fv = (*simStep.m_pForcesPrev)[uAtom];
        // copy the force information
        bool bDeferredCovalentBonds = false;
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            nvAssert(computeOutputForceOffset(u) == 7 + u);
            float nCovalentBonds = round(output.access(uAtom, 7 + u, 0, 0));
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
#endif
    }
#endif

    void copyConstAtomsDataFromTheModel(const std::vector<Atom<T>>& atoms)
    {
        GPUBuffer& dst = m_constAtomData;
        dst.resize<ConstantAtomData>(atoms.size());

        for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
        {
            const Atom<T>& srcAtom = atoms[uAtom];
            ConstantAtomData& dstAtom = dst.as<ConstantAtomData>(uAtom);
            dstAtom.fMass = (float)srcAtom.getMass();
            dstAtom.fValence = (float)srcAtom.getValence();
            dstAtom.fElectroNegativity = 1;
        }
    }
    std::shared_ptr<GPUBuffer> copyTransientAtomsDataFromTheModel(
        const std::vector<Atom<T>>& atoms) const
    {
        std::shared_ptr<GPUBuffer> pDst = std::make_shared<GPUBuffer>();
        GPUBuffer& dst = *pDst;
        dst.resize<TransientAtomData>(atoms.size());

        for (NvU32 uAtom = 0; uAtom < atoms.size(); ++uAtom)
        {
            const Atom<T>& srcAtom = atoms[uAtom];
            TransientAtomData& dstAtom = dst.as<TransientAtomData>(uAtom);
            copy(dstAtom.vPos, srcAtom.m_vPos);
            copy(dstAtom.vSpeed, srcAtom.m_vSpeed);
            dstAtom.fCharge = 0;
        }

        return pDst;
    }
    NvU32 findTheFarthestAtom(NvU32 uAtom, const ForceIndices& indices,
        const SimContext<T> &simContext) const
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
    std::shared_ptr<GPUBuffer> copyForceIndicesFromTheModel(
        const SimContext<T>& simContext) const
    {
        const ForceMap<T>& forceMap = simContext.m_forces;

        std::shared_ptr<GPUBuffer> pDst = std::make_shared<GPUBuffer>();
        GPUBuffer& pIndices = *pDst;
        pIndices.resize<ForceIndices>(getNAtoms());

        for (NvU32 uForce = 0; uForce < forceMap.size(); ++uForce)
        {
            if (!forceMap.isValid(uForce))
                continue;
            const Force<T>& force = forceMap.accessForceByIndex(uForce);
            {
                ForceIndices& fi1 = pIndices.as<ForceIndices>(force.getAtom1Index());
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
                ForceIndices& fi2 = pIndices.as<ForceIndices>(force.getAtom2Index());
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

        return pDst;
    }
    std::shared_ptr<GPUBuffer> copyBondsFromTheModel(const ForceMap<T>& forceMap,
        const GPUBuffer& pIndices) const
    {
        std::shared_ptr<GPUBuffer> pDst = std::make_shared<GPUBuffer>();
        GPUBuffer& pValues = *pDst;
        pValues.resize<ForceValues>(getNAtoms());

        for (NvU32 uAtom1 = 0; uAtom1 < getNAtoms(); ++uAtom1)
        {
            const ForceIndices& indices = pIndices.as<ForceIndices>(uAtom1);
            for (NvU32 u = 0; u < indices.nIndices; ++u)
            {
                NvU32 uAtom2 = indices.atomIndices[u];
                NvU32 uForce = forceMap.findExistingForce(uAtom1, uAtom2);
                const Force<T>& force = forceMap.accessForceByIndex(uForce);
                if (force.isCovalentBond())
                {
                    pValues.as<ForceValues>(uAtom1).m_nCovalentBonds[u] = 1;
                }
            }
        }

        return pDst;
    }

    GPUBuffer m_constAtomData; // 1 buffer - describes static properties of all simulated atoms
    struct Step
    {
        void serialize(ISerializer& s, Step *pPrevStep = nullptr)
        {
            s.serializeSimpleType("m_index", m_index);
            // can we save storage and get atoms state from the prev step
            if (pPrevStep && pPrevStep->m_index + 1 == m_index)
            {
                m_pAtomsPrev = pPrevStep->m_pAtomsNext;
            }
            else
            {
                s.serializeSharedPtr("m_pAtomsPrev", m_pAtomsPrev);
            }
            s.serializeSharedPtr("m_pAtomsNext", m_pAtomsNext);
            s.serializeSharedPtr("m_pForcesPrev", m_pForcesPrev);
            s.serializeSharedPtr("m_pForcesNext", m_pForcesNext);
            s.serializeSharedPtr("m_pForceIndices", m_pForceIndices);
        }
        NvU32 m_index = 0;
        std::shared_ptr<GPUBuffer> m_pAtomsPrev, m_pAtomsNext;
        std::shared_ptr<GPUBuffer> m_pForcesPrev, m_pForcesNext;
        std::shared_ptr<GPUBuffer> m_pForceIndices;
    };
    Step m_lastSimStep;
    std::vector<Step> m_simSteps;
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
};
