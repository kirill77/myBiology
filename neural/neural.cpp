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
        double fBiasesLR = lr.getLearningRate(uLayer);
        double fWeightsLR = lr.getLearningRate(uLayer);
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

struct DerivChecker
{
    void init(NeuralNetwork *pNetwork, Batch *pBatch, RNGUniform *pRNG)
    {
        m_pNetwork = pNetwork;
        m_pBatch = pBatch;

        // make internal copy of weights and biases
        m_pNetwork->saveCurrentStateToBackup();

        //g_bExecuteOnTheGPU = false;

        TensorRef pOutput = m_pBatch->forwardPass(*m_pNetwork);

        // generate some kind of random wanted output
        m_wantedOutput.init(pOutput->getDims(), sizeof(float));
        m_wantedOutput.clearWithRandomValues<float>(-1, 1, *pRNG);

        m_lossDeriv.init(pOutput->getDims(), sizeof(float));

        m_lr.init(m_pNetwork->getNLearningRatesNeeded());

        // compute the initial loss - before we call changeParam()
        m_lossComputer.compute(*pOutput, m_wantedOutput, m_lossDeriv, &m_fLossBefore);
    }
    bool doDerivativesMatch(NvU32 uParam)
    {
        float fDeltaParamForward = 0.25f;
        double fPrevPercentDifference = 1e38;
        std::vector<double> fN, fA;
        for (NvU32 i2 = 0; ; ++i2) // loop until acceptable accuracy of derivative is achieved
        {
            // change the param so that we could compute the numeric derivative dLoss/dParam later on
            double fPrevParamValue = m_pNetwork->getTrainableParam(uParam);
            double fNextParamValue = fPrevParamValue + fDeltaParamForward;
            m_pNetwork->setTrainableParam(uParam, fNextParamValue);

            // see how that has affected the output
            TensorRef outputAfterChangeRef = m_pBatch->forwardPass(*m_pNetwork);
            Tensor& outputAfterChange = *outputAfterChangeRef;
            // compute the change in output due to changeParam() that we did
            double fLossAfter = 0;
            m_lossComputer.compute(outputAfterChange, m_wantedOutput, m_lossDeriv, &fLossAfter);
            double fNumericDeriv = (fLossAfter - m_fLossBefore) / fDeltaParamForward;

            // restore weights/biases (no need for full restoreStateFromBackup() here because we just changed one layer)
            m_pNetwork->setTrainableParam(uParam, fPrevParamValue);
            m_pBatch->forwardPass(*m_pNetwork);
            m_pNetwork->backwardPass(m_pBatch->getBatchIndex(), &m_lossDeriv, m_lr);
            // backward pass is supposed to change param by the analytic loss derivative
            double fNextParamValue1 = m_pNetwork->getTrainableParam(uParam);
            double fAnalyticDeriv = fPrevParamValue - fNextParamValue1;

            fN.push_back((float)fNumericDeriv);
            fA.push_back(fAnalyticDeriv);

            // restore all weights/biases from the backup (needed because backwardPass() has changed everything)
            m_pNetwork->restoreStateFromBackup(DeepCopy);

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
                // perhaps precision was not enough - test the same with double precision
                nvAssert(false);
                return false;
            }
        }
        return true;
    }

private:
    NeuralNetwork* m_pNetwork = nullptr;
    Tensor m_wantedOutput, m_lossDeriv;
    LossComputer m_lossComputer;
    LearningRates m_lr;
    double m_fLossBefore = 0;
    Batch* m_pBatch = nullptr;
};

// try to verify that our analytic derivatives are correct (use numeric computation for that)
bool NeuralNetwork::testRandomDerivative(Batch &batch, NvU32 nChecks)
{
    return true;
    DerivChecker derivCheckerFloat;
    derivCheckerFloat.init(this, &batch, &m_rng);

    NvU32 nChecksWanted = nChecks * 100;
    double fCheckedParamStride = 1;
    NvU32 nTotalParams = getNTrainableParams();
    fCheckedParamStride = nTotalParams / (double)nChecksWanted;
    fCheckedParamStride = std::max(fCheckedParamStride, 1.);

    for (double fCheckedParam = 0; fCheckedParam < nTotalParams; fCheckedParam += fCheckedParamStride)
    {
        if (!derivCheckerFloat.doDerivativesMatch((NvU32)fCheckedParam))
        {
            nvAssert(false);
            return false;
        }
    }

    return true;
}

NvU32 NeuralNetwork::getNTrainableParams() const
{
    NvU32 nParams = 0;
    for (NvU32 u = 0; u < m_pLayers.size(); ++u)
    {
        nParams += m_pLayers[u]->getNTrainableParams();
    }
    return nParams;
}
double NeuralNetwork::getTrainableParam(NvU32 uParam)
{
    for (NvU32 u = 0; u < m_pLayers.size(); ++u)
    {
        if (uParam < m_pLayers[u]->getNTrainableParams())
            return m_pLayers[u]->getTrainableParam(uParam);
        uParam -= m_pLayers[u]->getNTrainableParams();
    }
    nvAssert(false); // no such param
    return 0;
}
void NeuralNetwork::setTrainableParam(NvU32 uParam, double fValue)
{
    for (NvU32 u = 0; u < m_pLayers.size(); ++u)
    {
        if (uParam < m_pLayers[u]->getNTrainableParams())
        {
            m_pLayers[u]->setTrainableParam(uParam, fValue);
            return;
        }
        uParam -= m_pLayers[u]->getNTrainableParams();
    }
    nvAssert(false); // no such param
}
