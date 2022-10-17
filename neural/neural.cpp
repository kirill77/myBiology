#include "network.h"
#include "learningRates.h"
#include "l2Computer.h"

std::shared_ptr<ILayer> ILayer::createLayer(LAYER_TYPE layerType)
{
    std::shared_ptr<ILayer> pLayer;
    switch (layerType)
    {
    case LAYER_TYPE_FCL_IDENTITY:
        pLayer = std::make_shared<FullyConnectedLayer<ACTIVATION_IDENTITY,
            ACTIVATION_IDENTITY>>();
        break;
    case LAYER_TYPE_FCL_MIRRORED:
        pLayer = std::make_shared<FullyConnectedLayer<ACTIVATION_RELU,
            ACTIVATION_MRELU>>();
        break;
    default:
        nvAssert(false);
        return nullptr;
    }
    nvAssert(pLayer->m_type == layerType);
    return pLayer;
}

void NeuralNetwork::backwardPass(NvU32 uBatch, Tensor<float>* pLoss, struct LearningRates& lr)
{
    NvU32 uLayer = (NvU32)m_pLayers.size() - 1;
    while (uLayer < m_pLayers.size())
    {
        float fBiasesLR = lr.getLearningRate(uLayer);
        float fWeightsLR = lr.getLearningRate(uLayer);
        m_fLRSum += fBiasesLR + fWeightsLR;
        m_nLRSamples += 2;
        pLoss = m_pLayers[uLayer]->backward(uBatch, *pLoss,
            fBiasesLR, fWeightsLR);
        --uLayer;
    }
}

TensorRef NeuralNetwork::getTmpTensor(TensorRef& pCache, const std::array<NvU32, 4>& dims)
{
    // if we have cached object and nobody is using this cache object except us and dimensions match - just return that object
    if (pCache && pCache.use_count() == 1 && pCache->getDims() == dims)
    {
        return pCache;
    }
    pCache = std::make_shared<Tensor<float>>(dims);
    return pCache;
}

// try to verify that our analytics derivatives are correct (use numeric computation for that)
bool NeuralNetwork::testRandomDerivative(Batch &batch, NvU32 nChecks)
{
    return true;
    // make internal copy of weights and biases
    saveCurrentStateToBackup();

    double fPrevErrorPercents = 1e38;

    // remember what the output was before we started changing weights/biases
    Tensor<float> outputBeforeChange;
    {
        TensorRef tmp = batch.forwardPass(*this);
        outputBeforeChange.init(tmp->getDims());
        outputBeforeChange.copySubregionFrom(0, *tmp, 0, tmp->size());
    }

    Tensor<float> loss;
    loss.init(outputBeforeChange.getDims());

    LearningRates lr;
    lr.init(getNLearningRatesNeeded());
    LossComputer lossComputer;

    for (NvU32 i1 = 0; i1 < nChecks; ++i1)
    {
        NvU32 uLayer = m_rng.generateUnsigned(0, (NvU32)m_pLayers.size());
        ILayer* pLayer = m_pLayers[uLayer].get();
        NvU32 nParams = pLayer->getNParams();
        NvU32 uParam = m_rng.generateUnsigned(0, nParams);
        float fDeltaParamForward = 0.25f;
        for (NvU32 i2 = 0; ; ++i2) // loop until acceptable accuracy of derivative is achieved
        {
            // change weight/bias by some small amount
            pLayer->changeParam(uParam, fDeltaParamForward);

            // see how that has affected the output
            TensorRef outputAfterChange = batch.forwardPass(*this);

            // compute the change in output due to changeParam() that we did
            lossComputer.compute(outputBeforeChange, *outputAfterChange, loss);

            // restore weights/biases
            pLayer->restoreStateFromBackup();

            // re-do the forward pass with the original value of parameter
            batch.forwardPass(*this);
            // now try to numerically calculate the change of parameters needed to change output by the loss we computed earlier
            batch.backwardPass(*this, loss, lr);

            float fDeltaParamBackward = pLayer->computeCurrentMinusBackup(uParam);
            double fErrorPercents = abs((fDeltaParamForward - fDeltaParamBackward) / (double)fDeltaParamForward) * 100;
            if (fErrorPercents > fPrevErrorPercents)
            {
                // we have decreased the step, but the error has increased? something is wrong
                nvAssert(false);
                return false;
            }
            if (fErrorPercents < 1) // error below threshold - ok
                break;
            fPrevErrorPercents = fErrorPercents;

            // error is above threshold - try smaller param change
            fDeltaParamForward /= 2;

            // restore all weights/biases from the backup (needed because backwardPass() has changed everything)
            restoreStateFromBackup();
        }
    }

    restoreStateFromBackup();
    return true;
}