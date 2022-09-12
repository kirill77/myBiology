#include"basics/mybasics.h"
#include "neuralTest.h"
#include "network.h"
#include "l2Computer.h"
#include "batchTrainer.h"

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
    virtual void initBatch(BatchTrainer& batchTrainer, NvU32 uBatch) override
    {
        nvAssert(uBatch < getNBatches());
        RNGUniform rng((uBatch + 1) * 0x12345);
        static const NvU32 NSAMPLES_PER_BATCH = 10;
        TensorRef pInput = std::make_shared<Tensor<float>>();
        pInput->init(NSAMPLES_PER_BATCH, s_inputDims[1], s_inputDims[2], s_inputDims[3]);
        pInput->clearWithRandomValues(0, 1, rng);
        TensorRef pWantedOutput = std::make_shared<Tensor<float>>();
        pWantedOutput->init(NSAMPLES_PER_BATCH, s_layer1OutputDims[1], s_layer1OutputDims[2], s_layer1OutputDims[3]);
        pWantedOutput->clearWithRandomValues(0, 1, rng);
        batchTrainer.init(*this, uBatch, pInput, pWantedOutput);
    }
private:
    std::array<unsigned, 4> s_inputDims = { 1, 4, 4, 1 };
    std::array<unsigned, 4> s_layer0OutputDims = { 1, 4, 3, 1 };
    std::array<unsigned, 4> s_layer1OutputDims = { 1, 1, 1, 1 };

    bool createTestLayers(std::vector<std::shared_ptr<ILayer>>& pLayers)
    {
        using Layer0Type = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
        std::shared_ptr<Layer0Type> pLayer0 = std::make_shared<Layer0Type>(0);
        pLayer0->init(s_inputDims, s_layer0OutputDims);
        pLayers.push_back(pLayer0);

        using Layer1Type = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        std::shared_ptr<Layer1Type> pLayer1 = std::make_shared<Layer1Type>(1);
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
    virtual void initBatch(BatchTrainer& batchTrainer, NvU32 uBatch) override
    {
        static const NvU32 NSAMPLES_PER_BATCH = 100;
        RNGUniform rng((uBatch + 1) * 0x12345);

        TensorRef pInput = std::make_shared<Tensor<float>>();
        Tensor<float>& input = *pInput;
        std::array<unsigned, 4> inputDims = s_layerDims[0];
        std::array<unsigned, 4> outputDims = *s_layerDims.rbegin();
        input.init(NSAMPLES_PER_BATCH, inputDims[1], inputDims[2], inputDims[3]);
        TensorRef pWantedOutput = std::make_shared<Tensor<float>>();
        Tensor<float>& wantedOutput = *pWantedOutput;
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

        batchTrainer.init(*this, pInput, pWantedOutput);
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

void NeuralTest::test()
{
    m_bTested = true;
    LossComputer lossComputer;

    {
        TestNetwork network;
        BatchTrainer batchTrainer;
        network.initBatch(batchTrainer, 0);
        for (; ; )
        {
            batchTrainer.makeMinimalProgress(network, lossComputer);
            if (batchTrainer.getLR().getNStepsMade() >= 10000)
                break;
        }
        float fError = batchTrainer.getLR().getLastError();
        m_bTested = m_bTested && fError > 2e-6 && fError < 2.6e-6;
        nvAssert(m_bTested);
    }

#if 0
    {
        Test1Network network;

        BatchTrainer batch0, batch1;
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
