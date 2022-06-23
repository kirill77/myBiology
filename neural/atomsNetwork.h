#pragma once

#include <memory>
#include "../basics/vectors.h"
#include "../basics/simContext.h"
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
    inline NvU32 getNAtoms() const { return (NvU32)m_constAtomData.size(); }
    virtual bool createLayers_impl() override
    {
        if (m_input.size() != 0) // already initialized?
            return true;

        NvU32 nSimulationSteps = (NvU32)m_pForceIndices.size();
        int nClusters = nSimulationSteps * getNAtoms();
        std::array<int, 4> inputDims = { nClusters, s_nInputValuesPerCluster, 1, 1 };
        // copy all data to input tensor
        for (NvU32 uStep = 0; uStep < nSimulationSteps; ++uStep)
        {
            copyStateToInputTensor(uStep);
            copyStateToOutputTensor(uStep);
        }
        std::array<int, 4> outputDims = { nClusters, s_nOutputValuesPerCluster, 1, 1 };

        using InternalLayerType = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
        for ( ; ; )
        {
            std::array<int, 4> internalOutDims = inputDims;
            internalOutDims[1] /= 2;
            internalOutDims[1] &= ~1; // must be even
            if (internalOutDims[1] <= outputDims[1]) break;
            m_pLayers.push_back(std::make_shared<InternalLayerType>(inputDims, internalOutDims));
            inputDims = internalOutDims;
        }
        using OutputLayerType = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        m_pLayers.push_back(std::make_shared<OutputLayerType>(inputDims, outputDims));

        return true;
    }
    void init(std::vector<Atom<T>>& atoms)
    {
        copyStateToGPU(atoms, m_constAtomData);
    }
    void notifyStepBeginning(const SimContext<T> &simContext)
    {
        if (hasEnoughData())
            return;
        m_bSimStepStarted = true;

        if (m_pTransientAtomData.size() == 0)
        {
            copyTransientAtomsDataFromTheModel(simContext.m_atoms);
        }
        copyForceIndicesFromTheModel(simContext);
        copyBondsFromTheModel(simContext.m_forces);
    }
    size_t sizeInBytes() const { return m_totalBytes; }
    void notifyStepDone(const SimContext<T> &simContext)
    {
        if (!m_bSimStepStarted)
            return;

        copyTransientAtomsDataFromTheModel(simContext.m_atoms);
        copyBondsFromTheModel(simContext.m_forces);

        m_bSimStepStarted = false;
    }
    bool hasEnoughData() const { return !m_bSimStepStarted && sizeInBytes() > 1024 * 1024 * 1; }
    void saveWeights();
    void loadWeights();
    void makePrediction();

private:
    static constexpr NvU32 computeNInputValuesPerCluster()
    {
        // input data per cluster consists of:
        NvU32 nInputBytesPerCluster = nAtomsPerCluster * (sizeof(ConstantAtomData) + sizeof(TransientAtomData));
        // this describes covalent bonds that the central atom has with the other atoms inside the cluster (-1 because it can't have bonds with itself)
        nInputBytesPerCluster += sizeof(float) * (nAtomsPerCluster - 1);
        nvAssert(nInputBytesPerCluster % sizeof(float) == 0);
        return nInputBytesPerCluster / sizeof(float);
    }
    static const NvU32 s_nInputValuesPerCluster = computeNInputValuesPerCluster();

    static constexpr NvU32 computeNOutputValuesPerCluster()
    {
        // output data per cluster consists of:
        NvU32 nOutputBytesPerCluster = sizeof(TransientAtomData);
        // this describes covalent bonds that the central atom has with the other atoms inside the cluster (-1 because it can't have bonds with itself)
        nOutputBytesPerCluster += sizeof(float) * (nAtomsPerCluster - 1);
        return nOutputBytesPerCluster / sizeof(float);
    }
    static const NvU32 s_nOutputValuesPerCluster = computeNOutputValuesPerCluster();

    void copyStateToInputTensor(NvU32 uSimStep)
    {
        clearTensorSubregionForSimStep(m_input, uSimStep);
        for (NvU32 uAtom = 0; uAtom < getNAtoms(); ++uAtom)
        {
            copyClusterToInputTensor(uSimStep, uAtom);
        }
    }
    void copyStateToOutputTensor(NvU32 uSimStep)
    {
        clearTensorSubregionForSimStep(m_wantedOutput, uSimStep);
        for (NvU32 uAtom = 0; uAtom < getNAtoms(); ++uAtom)
        {
            copyClusterToOutputTensor(uSimStep, uAtom);
        }
    }
    void clearTensorSubregionForSimStep(Tensor<float> &tensor, NvU32 uSimStep)
    {
        NvU32 uClusterIndex = uSimStep * getNAtoms();
        NvU32 uStartIndex = tensor.compute1DIndex(uClusterIndex, 0, 0, 0);
        NvU32 uEndIndex = tensor.compute1DIndex(uClusterIndex + getNAtoms(), 0, 0, 0);
        tensor.clearSubregion(uStartIndex, uEndIndex);
    }
    void copyClusterToInputTensor(NvU32 uSimStep, NvU32 uCluster)
    {
        NvU32 uClusterIndex = uSimStep * getNAtoms() + uCluster;
        NvU32 uDstIndex = m_input.compute1DIndex(uClusterIndex, 0, 0, 0);
#if ASSERT_ONLY_CODE
        NvU32 uDbgEndIndex = m_input.compute1DIndex(uClusterIndex + 1, 0, 0, 0);
#endif
        // copy the main atom of the cluster
        uDstIndex = copyAtomToTensor(m_input, uDstIndex, uSimStep, uCluster);
        // copy all other atoms of the cluster
        const ForceIndices<nAtomsPerCluster>& forceIndices = (*m_pForceIndices[uSimStep])[uCluster];
        const ForceValues<nAtomsPerCluster>& forceValues = (*m_pForceValues[uSimStep * 2])[uCluster];
        for (NvU32 u = 0; u < forceIndices.nIndices; ++u)
        {
            NvU32 uAtom = forceIndices.atomIndices[u];
            uDstIndex = copyAtomToTensor(m_input, uDstIndex, uSimStep, uAtom);
            m_input[uDstIndex++] = forceValues.m_nCovalentBonds[u];
        }
        nvAssert(uDstIndex < uDbgEndIndex);
    }
    void copyClusterToOutputTensor(NvU32 uSimStep, NvU32 uCluster)
    {
        NvU32 uClusterIndex = uSimStep * getNAtoms() + uCluster;
        NvU32 uDstIndex = m_wantedOutput.compute1DIndex(uClusterIndex, 0, 0, 0);
#if ASSERT_ONLY_CODE
        NvU32 uDbgEndIndex = m_wantedOutput.compute1DIndex(uClusterIndex + 1, 0, 0, 0);
#endif
        // copy the main atom of the cluster
        uDstIndex = copyAtomToTensor(m_wantedOutput, uDstIndex, uSimStep + 1, uCluster);
        // copy all other atoms of the cluster
        const ForceIndices<nAtomsPerCluster>& forceIndices = (*m_pForceIndices[uSimStep])[uCluster];
        const ForceValues<nAtomsPerCluster>& forceValues = (*m_pForceValues[uSimStep * 2 + 1])[uCluster];
        for (NvU32 u = 0; u < forceIndices.nIndices; ++u)
        {
            m_wantedOutput[uDstIndex++] = forceValues.m_nCovalentBonds[u];
        }
        nvAssert(uDstIndex < uDbgEndIndex);
    }
    NvU32 copyAtomToTensor(Tensor<float>& tensor, NvU32 dstOffset, NvU32 transientArrayIndex, NvU32 uAtom)
    {
        dstOffset = tensor.copySubregionFrom(dstOffset, m_constAtomData, uAtom, 1);
        dstOffset = tensor.copySubregionFrom(dstOffset, *m_pTransientAtomData[transientArrayIndex], uAtom, 1);
        return dstOffset;
    }
    void copyStateToGPU(const std::vector<Atom<T>>& atoms, GPUBuffer<ConstantAtomData> &dst)
    {
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
        m_totalBytes += dst.sizeInBytes();
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
        m_totalBytes += pIndices.sizeInBytes();

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
                // find the farthest atom
                NvU32 iFarthest = findTheFarthestAtom(force.getAtom1Index(), fi1, simContext);
                nvSwap(fi1.atomIndices[iFarthest], fi1.atomIndices[fi1.nIndices]);
            }
            if (fi2.nIndices >= fi2.atomIndices.size())
            {
                --fi2.nIndices;
                // find the farthest atom
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
        m_totalBytes += pValues.sizeInBytes();

        const GPUBuffer<ForceIndices<nAtomsPerCluster>>& pIndices = **m_pForceIndices.rbegin();
        for (NvU32 uAtom1 = 0; uAtom1 < getNAtoms(); ++uAtom1)
        {
            const ForceIndices<nAtomsPerCluster>& indices = pIndices[uAtom1];
            for (NvU32 u = 0; u < indices.nIndices; ++u)
            {
                NvU32 uAtom2 = indices.atomIndices[u];
                NvU32 uForce = forceMap.findExistingForceIndex(uAtom1, uAtom2);
                const Force<T>& force = forceMap.accessForceByIndex(uForce);
                if (force.isCovalentBond())
                {
                    pValues[uAtom1].m_nCovalentBonds[u] = 1;
                }
            }
        }
    }

    size_t m_totalBytes = 0; // used to limit the amount of training data
    bool m_bSimStepStarted = false;

    GPUBuffer<ConstantAtomData> m_constAtomData; // 1
    std::vector<GPUBuffer<TransientAtomData>*> m_pTransientAtomData; // 1 in the beginning + 1 per simulation step
    std::vector<GPUBuffer<ForceValues<nAtomsPerCluster>>*> m_pForceValues; // 2 per simulation step
    std::vector<GPUBuffer<ForceIndices<nAtomsPerCluster>>*> m_pForceIndices; // 1 per simulation step

    // those tensors are created from the arrays above
    Tensor<float> m_input;
    Tensor<float> m_wantedOutput;
};
