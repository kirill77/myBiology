#include "layer.h"

void ILayer::allocateBatchData(LayerBatchData& batchData, NvU32 n)
{
    std::array<unsigned, 4> outputDims = m_outputDims;
    outputDims[0] = n;

    std::vector<TensorRef>& deltaOutputs = batchData.m_deltaOutputs;
    deltaOutputs.resize(1);
    if (deltaOutputs[0] == nullptr)
    {
        deltaOutputs[0] = std::make_shared<Tensor<float>>();
    }
    deltaOutputs[0]->init(outputDims);

    std::vector<TensorRef>& outputs = batchData.m_outputs;
    outputs.resize(1);
    if (outputs[0] == nullptr)
    {
        outputs[0] = std::make_shared<Tensor<float>>();
    }
    outputs[0]->init(outputDims);
}