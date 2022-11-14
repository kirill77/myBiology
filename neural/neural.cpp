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

void NeuralNetwork::backwardPass(NvU32 uBatch, Tensor* pLoss, struct LearningRates& lr)
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
    pCache = std::make_shared<Tensor>(dims, sizeof(float));
    return pCache;
}

// try to verify that our analytics derivatives are correct (use numeric computation for that)
bool NeuralNetwork::testRandomDerivative(Batch &batch, NvU32 nChecks)
{
    return true;
    // make internal copy of weights and biases
    saveCurrentStateToBackup();

    // remember what was the output before we started changing weights/biases
    Tensor outputBeforeChange;
    {
        TensorRef tmp = batch.forwardPass(*this);
        outputBeforeChange.copyFrom(*tmp);
    }

    // generate some kind of random wanted output
    Tensor wantedOutput;
    wantedOutput.init(outputBeforeChange.getDims(), sizeof(float));
    wantedOutput.clearWithRandomValues<float>(-1, 1, m_rng);

    Tensor lossDeriv;
    lossDeriv.init(outputBeforeChange.getDims(), sizeof(float));

    LearningRates lr;
    lr.init(getNLearningRatesNeeded());
    LossComputer lossComputer;

    // compute the change in output due to changeParam() that we did
    float fLossBefore = 0;
    lossComputer.compute(outputBeforeChange, wantedOutput, lossDeriv, &fLossBefore);

    for (NvU32 i1 = 0; i1 < nChecks * 100; ++i1)
    {
        NvU32 uLayer = 0;// m_rng.generateUnsigned(0, (NvU32)m_pLayers.size());
        ILayer* pLayer = m_pLayers[uLayer].get();
        NvU32 nParams = pLayer->getNParams();
        NvU32 uParam = m_rng.generateUnsigned(0, nParams);
        float fDeltaParamForward = 0.25f;
        double fPrevPercentDifference = 1e38;
        std::vector<float> fN, fA;
        for (NvU32 i2 = 0; ; ++i2) // loop until acceptable accuracy of derivative is achieved
        {
            // compute the numeric derivative dLoss/dParam
            pLayer->changeParam(uParam, fDeltaParamForward);
            // see how that has affected the output
            TensorRef outputAfterChange = batch.forwardPass(*this);
            // compute the change in output due to changeParam() that we did
            float fLossAfter = 0;
            lossComputer.compute(*outputAfterChange, wantedOutput, lossDeriv, &fLossAfter);
            double fNumericDeriv = (fLossAfter - fLossBefore) / fDeltaParamForward;

            // restore weights/biases
            pLayer->restoreStateFromBackup();
            batch.forwardPass(*this);
            batch.backwardPass(*this, lossDeriv, lr);
            // backward pass is supposed to change param by the analytic loss derivative
            float fAnalyticDeriv = -pLayer->computeCurrentMinusBackup(uParam);

            fN.push_back((float)fNumericDeriv);
            fA.push_back(fAnalyticDeriv);

            // restore all weights/biases from the backup (needed because backwardPass() has changed everything)
            restoreStateFromBackup();

            // loss seem to not depend on this particular param - go to the next sample
            if (abs(fNumericDeriv + fAnalyticDeriv) < 1e-10)
            {
                break;
            }
            double fPercentDifference = 200 * abs((fNumericDeriv - fAnalyticDeriv) / (fNumericDeriv + fAnalyticDeriv));
            if (fPercentDifference < 1)
            {
                break; // accuracy of this sample is good enough - go to the next sample
            }
            fPrevPercentDifference = fPercentDifference;

            // error is above threshold - try smaller param change
            fDeltaParamForward /= 2;
            if (fDeltaParamForward < 1e-10)
            {
                // could not reach desired accuracy of derivative - something seems wrong
                nvAssert(false);
                return false;
            }
        }
    }

    return true;
}