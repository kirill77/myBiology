#include"basics/mybasics.h"
#include "neuralTest.h"
#include "network.h"

bool NeuralTest::m_bTested = false;

static std::array<unsigned, 4> s_inputDims({ 10, 4, 4, 1 });
static std::array<unsigned, 4> s_layer0OutputDims({ 10, 4, 3, 1 });
static std::array<unsigned, 4> s_layer1OutputDims({ 10, 1, 1, 1 });

struct TestNetwork : public NeuralNetwork
{
    virtual bool createLayers_impl(std::vector<std::shared_ptr<ILayer>> &pLayers) override
    {
        using Layer0Type = FullyConnectedLayer<ACTIVATION_RELU, ACTIVATION_MRELU>;
        std::shared_ptr<Layer0Type> pLayer0 = std::make_shared<Layer0Type>(s_inputDims, s_layer0OutputDims);
        pLayers.push_back(pLayer0);

        using Layer1Type = FullyConnectedLayer<ACTIVATION_IDENTITY, ACTIVATION_IDENTITY>;
        std::shared_ptr<Layer1Type> pLayer1 = std::make_shared<Layer1Type>(s_layer0OutputDims, s_layer1OutputDims);
        pLayers.push_back(pLayer1);
        return true;
    }
};

void NeuralTest::test()
{
    m_bTested = true;
    TestNetwork network;

    std::vector<TensorRef> inputs, wantedOutputs;
    {
        TensorRef input = std::make_shared<Tensor<float>>();
        input->init(s_inputDims);
        input->clearWithRandomValues(0, 1);
        inputs.push_back(input);
    }
    {
        TensorRef output = std::make_shared<Tensor<float>>();
        output->init(s_layer1OutputDims);
        output->clearWithRandomValues(0, 1);
        wantedOutputs.push_back(output);
    }

#if 1
    double fError = network.train(100, inputs, wantedOutputs);
    m_bTested = m_bTested && fError > 0.45 && fError < 0.55;
    nvAssert(m_bTested);
#endif
}
