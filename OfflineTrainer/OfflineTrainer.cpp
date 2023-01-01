#include <neural/atomsNetwork.h>
#include <chrono>
#include <filesystem>
#include "neural/l2Computer.h"
#include "neural/epoch.h"
#include "neural/learningRates.h"
#include "neural/neuralTest.h"

extern size_t g_nCudaBytes;

int main()
{
    NeuralTest::test();

    AtomsNetwork<float> network;

    // load the latest trained network
    {
        int nStepsMax = 0;
        std::filesystem::path path;
        for (const auto& entry : std::filesystem::directory_iterator("C:\\atomNets"))
        {
            int nSteps = 0;
            std::wstring sFileName = entry.path().filename();
            if (swscanf_s(sFileName.c_str(), L"trained_%d.bin", &nSteps) == 1)
            {
                if (nSteps > nStepsMax)
                {
                    nStepsMax = nSteps;
                    path = entry.path();
                }
            }
        }
        if (path.empty())
        {
            path = L"c:\\atomNets\\water_4236.bin";
        }
        printf("loading %S\n", path.c_str());
        MyReader reader(path);
        network.serialize(reader);
    }

    Epoch epoch;

    LossComputer lossComputer;
    lossComputer.init(sizeof(float));

    LearningRates lr;
    lr.init(network.getNLearningRatesNeeded());

    std::string sBuffer;
    sBuffer.resize(1024);

    printf("training starts...\n");
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;
    for ( ; ; )
    {
        epoch.makeStep(network, lossComputer, lr);

        auto curTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> secondsSinceStart = curTime - startTime;
        double fMSecsPerTrainingStep = (secondsSinceStart.count() / lr.getNStepsMade()) * 1000;

        sprintf_s(&sBuffer[0], sBuffer.size() - 2,
            "nSteps: %d, avgPreError: %#.3g, avgPostError: %#.3g, avgLRate: %#.3g, "
            "MSecPerStep: %.2f, MB: %.2f\n",
            lr.getNStepsMade(), epoch.getAvgPreError(), epoch.getAvgPostError(),
            network.computeAvgLRStats(), fMSecsPerTrainingStep, (double)g_nCudaBytes / (1024 * 1024));
        printf("%s", sBuffer.c_str());

        {
            FILE* fp = nullptr;
            fopen_s(&fp, "c:\\atomNets\\log.txt", "a+");
            if (fp)
            {
                fprintf(fp, "%s", sBuffer.c_str());
                fclose(fp);
            }
        }

        std::chrono::duration<double> secondsSinceLastSave = curTime - lastSaveTime;
        if (secondsSinceLastSave.count() > 60)
        {
            printf("saving nSteps=%d...\n", lr.getNStepsMade());
            wchar_t sBuffer[32];
            swprintf_s(sBuffer, L"c:\\atomNets\\trained_%d.bin",
                lr.getNStepsMade());
            {
                MyWriter writer(sBuffer);
                network.serialize(writer);
            }
            lastSaveTime = curTime;
            printf("saving completed\n");
        }

        network.resetAvgLRStats();
    }

#if 0
    BatchTrainer batchTrainer;


    network.initBatch(batchTrainer, 0);
    const NvU32 nLoadedTrainSteps = batchTrainer.getLR().getNStepsMade();

    if (nLoadedTrainSteps == 0)
    {
        // restart csv file from the beginning
        FILE* fp = nullptr;
        fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "wt");
        if (fp != nullptr)
        {
            fprintf(fp, "nSteps, fError, fLRate, MSecsPerStep\n");
            fclose(fp);
        }
        printf("starting training...\n");
    }
    else
    {
        printf("continuing previously saved training...\n");
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;

    const NvU32 nStepsPerCycle = 1024;
    for (NvU32 nCycles = 0; ; ++nCycles)
    {
        for ( ; ; )
        {
            batchTrainer.makeMinimalProgress(network, lossComputer);
            if (batchTrainer.getLR().getNStepsMade() >= (nCycles + 1) * nStepsPerCycle)
                break;
        }

        auto curTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> secondsInTraining = curTime - startTime;
        NvU32 nTrainStepsMadeThisSession = batchTrainer.getLR().getNStepsMade() - nLoadedTrainSteps;
        double fMSecsPerTrainingStep = (secondsInTraining.count() / nTrainStepsMadeThisSession) * 1000;

        double fAvgLRStats = network.computeAvgLRStats();
        network.resetAvgLRStats();
        {
            FILE* fp = nullptr;
            fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "a+");
            if (fp != nullptr)
            {
                fprintf(fp, "%d,  %#.3g,  %#.3g, %.2f\n",
                    batchTrainer.getLR().getNStepsMade(), batchTrainer.getLR().getLastError(),
                    fAvgLRStats, fMSecsPerTrainingStep);
                fclose(fp);
            }
        }

        std::chrono::duration<double> secondsSinceLastSave = curTime - lastSaveTime;
        if (secondsSinceLastSave.count() > 60)
        {
            printf("saving nSteps=%d...\n", batchTrainer.getLR().getNStepsMade());
            wchar_t sBuffer[32];
            swprintf_s(sBuffer, L"c:\\atomNets\\trained_%d.bin",
                batchTrainer.getLR().getNStepsMade());
            {
            MyWriter writer(sBuffer);
            network.serialize(writer);
            batchTrainer.serialize(writer);
            }
            lastSaveTime = curTime;
            printf("saving completed\n");
        }

        printf("nSteps: %d, fError: %#.3g, fLRate: %#.3g, MSecsPerStep: %.2f\n",
            batchTrainer.getLR().getNStepsMade(), batchTrainer.getLR().getLastError(), fAvgLRStats, fMSecsPerTrainingStep);
    }
#endif
}
