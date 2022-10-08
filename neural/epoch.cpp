#include "epoch.h"
#include "network.h"
#include "learningRates.h"

void Epoch::makeStep(NeuralNetwork &network, LossComputer &lossComputer, LearningRates &lr)
{
    printf("0%%\r");
    NvU32 uPrevPercent = 0;

    double fPreErrorsSum = 0, fPostErrorsSum = 0;
    NvU32 nBatches = network.getNBatches();
    for (NvU32 uBatch = 0; uBatch < nBatches; ++uBatch)
    {
        Batch batch = network.allocateBatchData(uBatch);
        fPreErrorsSum += batch.makeMinimalProgress(network, lossComputer, lr);
        fPostErrorsSum += lr.getLastError();

        NvU32 uNextPercent = uBatch * 100 / (NvU32)nBatches;
        if (uNextPercent > uPrevPercent + 5)
        {
            printf("%d%%\r", uNextPercent);
            uPrevPercent = uNextPercent;
        }
    }
    m_fAvgPreError = (float)(fPreErrorsSum / nBatches);
    m_fAvgPostError = (float)(fPostErrorsSum / nBatches);
}