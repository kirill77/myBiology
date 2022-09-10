#include"basics/mybasics.h"
#include "neuralTest.h"
#include "network.h"

bool NeuralTest::m_bTested = false;

static std::array<unsigned, 4> s_inputDims({ 1, 4, 4, 1 });
static std::array<unsigned, 4> s_layer0OutputDims({ 1, 4, 3, 1 });
static std::array<unsigned, 4> s_layer1OutputDims({ 1, 1, 1, 1 });

struct TestNetwork : public NeuralNetwork
{
    bool createTestLayers(std::vector<std::shared_ptr<ILayer>> &pLayers)
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
    TestNetwork()
    {
        createTestLayers(m_pLayers);
    }
    virtual NvU32 getNBatches() override
    {
        return 4;
    }
    virtual void initBatch(BatchTrainer& batchTrainer, NvU32 uBatch)
    {
        nvAssert(uBatch < getNBatches());
        RNGUniform rng((uBatch + 1) * 0x12345);
        static const NvU32 NSAMPLES_PER_BATCH = 10;
        std::vector<TensorRef> inputs, wantedOutputs;
        {
            TensorRef input = std::make_shared<Tensor<float>>();
            input->init(NSAMPLES_PER_BATCH, s_inputDims[1], s_inputDims[2], s_inputDims[3]);
            input->clearWithRandomValues(0, 1, rng);
            inputs.push_back(input);
        }
        {
            TensorRef output = std::make_shared<Tensor<float>>();
            output->init(NSAMPLES_PER_BATCH, s_layer1OutputDims[1], s_layer1OutputDims[2], s_layer1OutputDims[3]);
            output->clearWithRandomValues(0, 1, rng);
            wantedOutputs.push_back(output);
        }
        batchTrainer.init(*this, inputs, wantedOutputs);
    }
};

void NeuralTest::test()
{
    m_bTested = true;
    TestNetwork network;
    LossComputer lossComputer;

#if 1
    BatchTrainer batchTrainer;
    network.initBatch(batchTrainer, 0);
    for ( ; ; )
    {
        batchTrainer.makeMinimalProgress(network, lossComputer);
        if (batchTrainer.getLR().getNStepsMade() >= 10000)
            break;
    }
    float fError = batchTrainer.getLR().getLastError();
    m_bTested = m_bTested && fError > 2e-6 && fError < 2.6e-6;
    nvAssert(m_bTested);
#endif
}
