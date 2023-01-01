#include"basics/mybasics.h"
#include "neuralTest.h"
#include "network.h"
#include "l2Computer.h"
#include "batch.h"
#include "learningRates.h"

bool NeuralTest::m_bTested = false;

struct TestNetwork : public NeuralNetwork
{
    TestNetwork()
    {
        createTestLayers(m_pLayers);
    }
    virtual NvU32 getNBatches() override
    {
        return 1;
    }
    virtual std::shared_ptr<Batch> createAndInitBatchInternal(NvU32 uBatch) override
    {
        nvAssert(uBatch < getNBatches());
        RNGUniform rng((uBatch + 1) * 0x12345);
        static const NvU32 NSAMPLES_PER_BATCH = 10;
        TensorRef pInput = std::make_shared<Tensor>(NSAMPLES_PER_BATCH, s_inputDims[1], s_inputDims[2], s_inputDims[3], sizeof(float));
        pInput->clearWithRandomValues<float>(0, 1, rng);
        TensorRef pWantedOutput = std::make_shared<Tensor>(NSAMPLES_PER_BATCH, s_layer1OutputDims[1], s_layer1OutputDims[2], s_layer1OutputDims[3],         sizeof(float));
        pWantedOutput->clearWithRandomValues<float>(0, 1, rng);
        return std::make_shared<Batch>(uBatch, pInput, pWantedOutput);
    }

    virtual std::shared_ptr<NeuralNetwork> cloneToPrecision(NvU32 elemSize) override
    {
        std::shared_ptr<TestNetwork> p = std::make_shared<TestNetwork>(*this);
        for (NvU32 u = 0; u < m_pLayers.size(); ++u)
        {
            p->m_pLayers[u] = m_pLayers[u]->cloneToPrecision(elemSize);
        }
        return p;
    }

private:
    std::array<unsigned, 4> s_inputDims = { 1, 4, 4, 1 };
    std::array<unsigned, 4> s_layer0OutputDims = { 1, 4, 3, 1 };
    std::array<unsigned, 4> s_layer1OutputDims = { 1, 1, 1, 1 };

    bool createTestLayers(std::vector<std::shared_ptr<ILayer>>& pLayers)
    {
        using Layer0Type = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
        std::shared_ptr<Layer0Type> pLayer0 = std::make_shared<Layer0Type>();
        pLayer0->init(s_inputDims, s_layer0OutputDims);
        pLayers.push_back(pLayer0);

        using Layer1Type = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        std::shared_ptr<Layer1Type> pLayer1 = std::make_shared<Layer1Type>();
        pLayer1->init(s_layer0OutputDims, s_layer1OutputDims);
        pLayers.push_back(pLayer1);
        return true;
    }
};

#if 0
struct Test1Network : public NeuralNetwork
{
    Test1Network()
    {
        createTestLayers(m_pLayers);
    }
    virtual NvU32 getNBatches() override
    {
        return 1;
    }
    virtual void initBatch(Batch& batch, NvU32 uBatch) override
    {
        static const NvU32 NSAMPLES_PER_BATCH = 100;
        RNGUniform rng((uBatch + 1) * 0x12345);

        TensorRef pInput = std::make_shared<Tensor>();
        Tensor& input = *pInput;
        std::array<unsigned, 4> inputDims = s_layerDims[0];
        std::array<unsigned, 4> outputDims = *s_layerDims.rbegin();
        input.init(NSAMPLES_PER_BATCH, inputDims[1], inputDims[2], inputDims[3]);
        TensorRef pWantedOutput = std::make_shared<Tensor>();
        Tensor& wantedOutput = *pWantedOutput;
        wantedOutput.init(NSAMPLES_PER_BATCH, outputDims[1], outputDims[2], outputDims[3]);

        for (int i = 0; i < (int)input.n(); ++i)
        {
            rtvector<float, 3> p1, p2;
            for (NvU32 uDim = 0; uDim < 3; ++uDim)
            {
                p1[uDim] = (float)(rng.generate01() * 10 - 5);
                p2[uDim] = (float)(rng.generate01() * 10 - 5);
                input.access(i, uDim * 2, 0, 0) = p1[uDim];
                input.access(i, uDim * 2 + 1, 0, 0) = p2[uDim];
            }
            float fDist = length(p1 - p2);
            wantedOutput.access(i, 0, 0, 0) = sin(fDist);
        }

        batch.init(*this, pInput, pWantedOutput);
    }

private:
    std::array<std::array<unsigned, 4>, 3> s_layerDims = { { { 1, 6, 1, 1 }, { 1, 64, 1, 1 }, { 1, 1, 1, 1 } } };

    void createTestLayers(std::vector<std::shared_ptr<ILayer>>& pLayers)
    {
        // create hidden layers
        using HiddenType = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
        for (int i = 0; i < s_layerDims.size() - 2; ++i)
        {
            std::shared_ptr<HiddenType> pHiddenLayer = std::make_shared<HiddenType>((NvU32)m_pLayers.size());
            pHiddenLayer->init(s_layerDims[i], s_layerDims[i + 1]);
            m_pLayers.push_back(pHiddenLayer);
        }

        // create output layer
        using OutputType = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        std::shared_ptr<OutputType> pOutputLayer = std::make_shared<OutputType>((NvU32)m_pLayers.size());
        pOutputLayer->init(s_layerDims.rbegin()[1], s_layerDims.rbegin()[0]);
        m_pLayers.push_back(pOutputLayer);
    }
};
#endif

struct DerivChecker
{
    void init(std::shared_ptr<NeuralNetwork> pNetwork, std::shared_ptr<Batch> pBatch, RNGUniform* pRNG)
    {
        m_pNetwork = pNetwork;
        m_pBatch = pBatch;

        // make internal copy of weights and biases
        m_pNetwork->saveCurrentStateToBackup();

        //g_bExecuteOnTheGPU = false;

        TensorRef pOutput = m_pBatch->forwardPass(*m_pNetwork);

        // generate some kind of random wanted output if we don't have them yet
        if (m_pWantedOutput == nullptr)
        {
            m_pWantedOutput = std::make_shared<Tensor>(pOutput->getDims(), sizeof(float));
            m_pWantedOutput->clearWithRandomValues<float>(-1, 1, *pRNG);
        }

        m_pLossDeriv = std::make_shared<Tensor>(pOutput->getDims(), m_pWantedOutput->elemSize());

        m_lr.init(m_pNetwork->getNLearningRatesNeeded());

        // compute the initial loss - before we call changeParam()
        m_lossComputer.init(m_pWantedOutput->elemSize());
        m_lossComputer.compute(*pOutput, *m_pWantedOutput, *m_pLossDeriv, &m_fLossBefore);
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
            m_lossComputer.compute(outputAfterChange, *m_pWantedOutput, *m_pLossDeriv, &fLossAfter);
            double fNumericDeriv = (fLossAfter - m_fLossBefore) / fDeltaParamForward;

            // restore weights/biases (no need for full restoreStateFromBackup() here because we just changed one layer)
            m_pNetwork->setTrainableParam(uParam, fPrevParamValue);
            m_pBatch->forwardPass(*m_pNetwork);
            m_pNetwork->backwardPass(m_pBatch->getBatchIndex(), m_pLossDeriv.get(), m_lr);
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
                // even with double precision couldn't reach the desired accuracy? something seems wrong
                nvAssert(m_pWantedOutput->elemSize() == 4);
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<DerivChecker> cloneToPrecision(NvU32 elemSize)
    {
        std::shared_ptr<DerivChecker> p = std::make_shared<DerivChecker>();
        auto pNetwork = m_pNetwork->cloneToPrecision(elemSize);
        auto pBatch = m_pBatch->cloneToPrecision(elemSize);
        p->m_pWantedOutput = m_pWantedOutput->cloneToPrecision(elemSize);
        p->init(pNetwork, pBatch, nullptr);
        return p;
    }

private:
    std::shared_ptr<NeuralNetwork> m_pNetwork;
    TensorRef m_pWantedOutput, m_pLossDeriv;
    LossComputer m_lossComputer;
    LearningRates m_lr;
    double m_fLossBefore = 0;
    std::shared_ptr<Batch> m_pBatch;
};

bool NeuralTest::testRandomDerivative(std::shared_ptr<NeuralNetwork> pNetwork, std::shared_ptr<Batch> pBatch, NvU32 nChecks)
{
    RNGUniform rng;
    std::shared_ptr<DerivChecker> m_pDerivCheckerF = std::make_shared<DerivChecker>();
    m_pDerivCheckerF->init(pNetwork, pBatch, &rng);
    std::shared_ptr<DerivChecker> m_pDerivCheckerD = m_pDerivCheckerF->cloneToPrecision(sizeof(double));

    NvU32 nChecksWanted = nChecks * 100;
    double fCheckedParamStride = 1;
    NvU32 nTotalParams = pNetwork->getNTrainableParams();
    fCheckedParamStride = nTotalParams / (double)nChecksWanted;
    fCheckedParamStride = std::max(fCheckedParamStride, 1.);

    for (double fCheckedParam = 0; fCheckedParam < nTotalParams; fCheckedParam += fCheckedParamStride)
    {
        if (!m_pDerivCheckerF->doDerivativesMatch((NvU32)fCheckedParam))
        {
            if (!m_pDerivCheckerD->doDerivativesMatch((NvU32)fCheckedParam)) // try with double precision
            {
                nvAssert(false);
                return false;
            }
        }
    }

    return true;
}

void NeuralTest::test()
{
    m_bTested = true;
    LossComputer lossComputer;
    lossComputer.init(sizeof(float));

    {
        std::shared_ptr<TestNetwork> pNetwork;
        pNetwork = std::make_shared<TestNetwork>();
        
        LearningRates lr;
        lr.init(pNetwork->getNLearningRatesNeeded());
        
        std::shared_ptr<Batch> pBatch = pNetwork->createBatch(0);
        for ( ; ; )
        {
            pBatch->makeMinimalProgress(*pNetwork, lossComputer, lr);
            if (lr.getNStepsMade() >= 10000)
                break;
        }
        double fError = lr.getLastError();
        m_bTested = m_bTested && fError > 0 && fError < 1e-11;
        nvAssert(m_bTested);

        m_bTested = m_bTested && testRandomDerivative(pNetwork, pBatch, 100);
        nvAssert(m_bTested);
    }

#if 0
    {
        Test1Network network;

        Batch batch0, batch1;
        network.initBatch(batch0, 0);
        network.initBatch(batch1, 1);

        batch0.makeMinimalProgress(network, lossComputer);
        float fError0 = batch0.getLR().getLastError();
        batch1.makeMinimalProgress(network, lossComputer);
        float fError1 = batch1.getLR().getLastError();

        for (; ; )
        {
            batch0.makeMinimalProgress(network, lossComputer);
            if (batch0.getLR().getNStepsMade() >= 10000)
                break;
        }
        for (; ; )
        {
            batch1.makeMinimalProgress(network, lossComputer);
            if (batch1.getLR().getNStepsMade() >= 10000)
                break;
        }
        {
            TextWriter writer("c:\\atomNets\\network1.txt");
            network.serialize(writer);
        }

        float fError01 = 0, fError11 = 0;
        for (NvU32 u = 0; u < 10; ++u)
        {
            batch0.makeMinimalProgress(network, lossComputer);
            fError01 = batch0.getLR().getLastError();
            batch1.makeMinimalProgress(network, lossComputer);
            fError11 = batch1.getLR().getLastError();
        }

        nvAssert(m_bTested);
    }
#endif
}
