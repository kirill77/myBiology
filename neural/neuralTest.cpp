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
        batchTrainer.init(*this, pInput, pWantedOutput);
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
        Tensor<float> input;
    }

private:
    void createTestLayers(std::vector<std::shared_ptr<ILayer>>& pLayers)
    {

    }
};

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
}
