#include "epoch.h"
#include "network.h"
#include "learningRates.h"

void Epoch::init(NeuralNetwork& network)
{
    NvU32 nBatches = network.getNBatches();
    m_batches.resize(nBatches);
    printf("creating batches...\n");
    for (NvU32 uBatch = 0; uBatch < nBatches; ++uBatch)
    {
        network.initBatch(m_batches[uBatch], uBatch);
    }
    printf("done creating batches\n");
}

void Epoch::makeStep(NeuralNetwork &network, LossComputer &lossComputer, LearningRates &lr)
{
    printf("0%%\r");
    NvU32 uPrevPercent = 0;

    double fPreErrorsSum = 0, fPostErrorsSum = 0;

    for (NvU32 u = 0; u < m_batches.size(); ++u)
    {
        fPreErrorsSum += m_batches[u].makeMinimalProgress(network, lossComputer, lr);
        fPostErrorsSum += lr.getLastError();

        NvU32 uNextPercent = u * 100 / (NvU32)m_batches.size();
        if (uNextPercent > uPrevPercent + 5)
        {
            printf("%d%%\r", uNextPercent);
            uPrevPercent = uNextPercent;
        }
    }
    m_fAvgPreError = (float)(fPreErrorsSum / m_batches.size());
    m_fAvgPostError = (float)(fPostErrorsSum / m_batches.size());
}