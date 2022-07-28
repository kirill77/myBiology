#pragma once

#include <memory>
#include "../basics/vectors.h"
#include "../basics/simContext.h"
#include "../basics/serializer.h"
#include "network.h"

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

template <class T, NvU32 nAtomsPerCluster>
struct AtomsNetwork : public NeuralNetwork
{
    using InternalLayerType = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;

    inline NvU32 getNAtoms() const { return (NvU32)m_constAtomData.size(); }
    double trainAtomsNetwork(NvU32 nSteps)
    {
        if (getNCompletedTrainSteps() == 0)// % NATOMS_IN_TRAINING == 0)
        {
            initializeTrainingData();
        }
        return train(nSteps, m_inputs, m_wantedOutputs);
    }
    void init(std::vector<Atom<T>>& atoms)
    {
        copyConstAtomsDataFromTheModel(atoms);
    }
    void notifyStepBeginning(const SimContext<T> &simContext)
    {
        if (hasEnoughData())
        {
            if (m_bNeedToSave)
            {
                m_bNeedToSave = false;
                MyWriter writer("atomsNetwork.bin");
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
    void loadFromFile(const std::string& sFileName)
    {
        MyReader s(sFileName);
        serialize(s);
    }

private:
    virtual bool createLayers_impl(std::vector<std::shared_ptr<ILayer>>& pLayers) override
    {
        nvAssert(pLayers.size() == 0);

        std::array<unsigned, 4> prevOutputDims = m_inputs[0]->getDims();
        std::array<unsigned, 4> idealOutputDims = prevOutputDims;
        std::array<unsigned, 4> wantedOutputDims = m_wantedOutputs[0]->getDims();
        for (; ; )
        {
            NvU32 nLayers = (NvU32)pLayers.size();
            for (NvU32 uDim = 0; uDim < idealOutputDims.size(); ++uDim)
            {
                if (idealOutputDims[uDim] / 2 <= wantedOutputDims[uDim])
                    continue;
                idealOutputDims[uDim] /= 2;

                std::array<unsigned, 4> outputDims = idealOutputDims;
                // to avoid dying neuron problem, we duplicate and mirror all neurons in H direction
                outputDims[1] *= 2;
                // to get nice alignment to CUDA warps - align H up to 32
                outputDims[1] = ((outputDims[1] + 31) / 32) * 32;
                pLayers.push_back(std::make_shared<InternalLayerType>(prevOutputDims, outputDims));
                prevOutputDims = outputDims;
                break;
            }
            if (nLayers == pLayers.size()) // no new layers created? we're done then
                break;
        }
        using OutputLayerType = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        pLayers.push_back(std::make_shared<OutputLayerType>(prevOutputDims, wantedOutputDims));
        return true;
    }
    static const NvU32 NATOMS_IN_TRAINING = 256; // number of atoms we train on simultaneously
    void initializeTrainingData()
    {
        NvU32 nSimulationSteps = (NvU32)m_pForceIndices.size();
        NvU32 nTotalClusters = nSimulationSteps * getNAtoms();
        // out of all clusters, randomly select some number of clusters we'll train on
        std::array<NvU32, NATOMS_IN_TRAINING> clusterIndices;
        for (NvU32 u = 0; u < clusterIndices.size(); ++u)
        {
            clusterIndices[u] = m_rng.generateUnsigned(0, nTotalClusters);
        }
        for (NvU32 nPasses = 0; nPasses < 10; ++nPasses) // we have this many tries to get the array without repetitions
        {
            bool bRepetitionsFound = false;
            std::sort(clusterIndices.begin(), clusterIndices.end());
            for (NvU32 u = 1; u < clusterIndices.size(); ++u)
            {
                while (clusterIndices[u] == clusterIndices[u - 1])
                {
                    clusterIndices[u] = m_rng.generateUnsigned(0, nTotalClusters);
                    bRepetitionsFound = true;
                }
            }
            if (!bRepetitionsFound) break;
        }

        if (m_inputs.size() == 0)
        {
            TensorRef input = std::make_shared<Tensor<float>>();
            m_inputs.push_back(input);
            TensorRef wantedOutput = std::make_shared<Tensor<float>>();
            m_wantedOutputs.push_back(wantedOutput);
        }

        // initialize inputs and outputs to zero (force CPU because we have to fill those buffers on CPU)
        std::array<unsigned, 4> inputDims = { (NvU32)clusterIndices.size(), s_nInputValuesPerCluster, 1, 1 };
        m_inputs[0]->init(inputDims);
        m_inputs[0]->clearSubregion(0, (NvU32)m_inputs[0]->size(), EXECUTE_MODE_FORCE_CPU);
        std::array<unsigned, 4> outputDims = { (NvU32)clusterIndices.size(), s_nOutputValuesPerCluster, 1, 1 };
        m_wantedOutputs[0]->init(outputDims);
        m_wantedOutputs[0]->clearSubregion(0, (NvU32)m_wantedOutputs[0]->size(), EXECUTE_MODE_FORCE_CPU);

        // copy all data to input and output tensors
        for (NvU32 u = 0; u < clusterIndices.size(); ++u)
        {
            copyClusterToInputTensor(u, clusterIndices[u]);
            copyClusterToOutputTensor(u, clusterIndices[u]);
        }
    }
    // **** input offsets computation
    static constexpr NvU32 computeInputAtomOffset(NvU32 uAtom)
    {
        NvU32 inputAtomSizeInBytes = sizeof(ConstantAtomData) + sizeof(TransientAtomData);
        nvAssert(inputAtomSizeInBytes % sizeof(float) == 0);
        return uAtom * (inputAtomSizeInBytes / sizeof(float));
    }
    static constexpr NvU32 computeInputForceOffset(NvU32 uForce)
    {
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
        return computeOutputAtomOffset(1) + uForce; // we output just one atom for now
    }
    // this is essentially an offset to the force one after the last
    static const NvU32 s_nOutputValuesPerCluster = computeOutputForceOffset(nAtomsPerCluster);

    void copyClusterToInputTensor(NvU32 uDstCluster, NvU32 uSrcCluster)
    {
        NvU32 uSimStep = uSrcCluster / getNAtoms();
        NvU32 uCentralAtom = uSrcCluster % getNAtoms();

        const GPUBuffer<TransientAtomData>& transientData = *m_pTransientAtomData[uSimStep];
        const ForceIndices<nAtomsPerCluster>& fi = (*m_pForceIndices[uSimStep])[uCentralAtom];
        const ForceValues<nAtomsPerCluster>& fv = (*m_pForceValues[uSimStep * 2])[uCentralAtom];

        copyAtomToInputTensor(uDstCluster, nAtomsPerCluster - 1, uCentralAtom, uCentralAtom, transientData); // copy the central atom
        Tensor<float>& m_input = *m_inputs[0];
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            copyAtomToInputTensor(uDstCluster, u, uCentralAtom, fi.atomIndices[u], transientData); // copy auxiliary atoms
            m_input.access(uDstCluster, computeInputForceOffset(u), 0, 0) = fv.m_nCovalentBonds[u]; // copy the force information
        }
    }
    void copyAtomToInputTensor(NvU32 uDstCluster, NvU32 uDstSlot, NvU32 uCentralAtom, NvU32 uSrcAtom, const GPUBuffer<TransientAtomData> &transientData)
    {
        NvU32 hi = computeInputAtomOffset(uDstSlot);
        const ConstantAtomData& constData = m_constAtomData[uSrcAtom];
        const TransientAtomData& transData = transientData[uSrcAtom];
        const TransientAtomData& centData = transientData[uCentralAtom];
        Tensor<float>& m_input = *m_inputs[0];
        m_input.access(uDstCluster, hi++, 0, 0) = constData.fElectroNegativity;
        m_input.access(uDstCluster, hi++, 0, 0) = constData.fMass;
        m_input.access(uDstCluster, hi++, 0, 0) = constData.fValence;
        m_input.access(uDstCluster, hi++, 0, 0) = transData.fCharge;
        rtvector<float, 3> vPos = transData.vPos - centData.vPos;
        m_input.access(uDstCluster, hi++, 0, 0) = vPos[0];
        m_input.access(uDstCluster, hi++, 0, 0) = vPos[1];
        m_input.access(uDstCluster, hi++, 0, 0) = vPos[2];
        rtvector<float, 3> vSpeed = transData.vSpeed - centData.vSpeed;
        m_input.access(uDstCluster, hi++, 0, 0) = vSpeed[0];
        m_input.access(uDstCluster, hi++, 0, 0) = vSpeed[1];
        m_input.access(uDstCluster, hi++, 0, 0) = vSpeed[2];
    }
    void copyClusterToOutputTensor(NvU32 uDstCluster, NvU32 uSrcCluster)
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
        NvU32 hi = 0;
        rtvector<float, 3> vPos = centAtomOut.vPos - centAtomIn.vPos;
        Tensor<float>& m_output = *m_wantedOutputs[0];
        m_output.access(uDstCluster, hi++, 0, 0) = vPos[0];
        m_output.access(uDstCluster, hi++, 0, 0) = vPos[1];
        m_output.access(uDstCluster, hi++, 0, 0) = vPos[2];
        rtvector<float, 3> vSpeed = centAtomOut.vSpeed - centAtomIn.vSpeed;
        m_output.access(uDstCluster, hi++, 0, 0) = vSpeed[0];
        m_output.access(uDstCluster, hi++, 0, 0) = vSpeed[1];
        m_output.access(uDstCluster, hi++, 0, 0) = vSpeed[2];

        // copy the force information
        for (NvU32 u = 0; u < fi.nIndices; ++u)
        {
            m_output.access(uDstCluster, computeOutputForceOffset(u), 0, 0) = fv.m_nCovalentBonds[u];
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
        NvU32 uFarthestIndex = -1;
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
            ForceIndices<nAtomsPerCluster>& fi1 = pIndices[force.getAtom1Index()];
            ForceIndices<nAtomsPerCluster>& fi2 = pIndices[force.getAtom2Index()];
            if (fi1.nIndices >= fi1.atomIndices.size())
            {
                --fi1.nIndices;
                // find the farthest atom and place it at the last position (it will be replaced)
                NvU32 iFarthest = findTheFarthestAtom(force.getAtom1Index(), fi1, simContext);
                nvSwap(fi1.atomIndices[iFarthest], fi1.atomIndices[fi1.nIndices]);
            }
            if (fi2.nIndices >= fi2.atomIndices.size())
            {
                --fi2.nIndices;
                // find the farthest atom and place it at the last position (it will be replaced)
                NvU32 iFarthest = findTheFarthestAtom(force.getAtom2Index(), fi2, simContext);
                nvSwap(fi2.atomIndices[iFarthest], fi2.atomIndices[fi2.nIndices]);
            }
            fi1.atomIndices[fi1.nIndices++] = force.getAtom2Index();
            fi2.atomIndices[fi2.nIndices++] = force.getAtom1Index();
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

    void serialize(ISerializer &s)
    {
        m_constAtomData.serialize(s);
        s.serialize(m_pTransientAtomData);
        s.serialize(m_pForceValues);
        s.serialize(m_pForceIndices);
    }

    GPUBuffer<ConstantAtomData> m_constAtomData; // 1 buffer - describes static properties of all simulated atoms
    std::vector<GPUBuffer<TransientAtomData>*> m_pTransientAtomData; // 1 in the beginning + 1 per simulation step
    std::vector<GPUBuffer<ForceValues<nAtomsPerCluster>>*> m_pForceValues; // 2 per simulation step
    std::vector<GPUBuffer<ForceIndices<nAtomsPerCluster>>*> m_pForceIndices; // 1 per simulation step

    // those tensors are created from the arrays above
    std::vector<TensorRef> m_inputs, m_wantedOutputs;

    RNGUniform m_rng;
};
