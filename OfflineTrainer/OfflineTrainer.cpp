#include <neural/atomsNetwork.h>
#include <chrono>
#include <filesystem>
#include "neural/l2Computer.h"
#include "neural/epoch.h"
#include "neural/learningRates.h"
#include "neural/neuralTest.h"
#include "offlineTrainer.h"

extern size_t g_nCudaBytes;

int OfflineTrainer::startTraining()
{
    NeuralTest::test();

    AtomsDataLoader<float> loader;

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
            path = L"c:\\atomNets\\water_4238.bin";
        }
        printf("loading %S\n", path.c_str());
        MyReader reader(path);
        loader.serialize(reader);
    }

    std::shared_ptr<NeuralNetwork> pNetwork = loader.createNetwork();

    Epoch epoch;

    LossComputer lossComputer;
    lossComputer.init(sizeof(float));

    LearningRates lr;
    lr.init(pNetwork->getNLearningRatesNeeded());

    std::string sBuffer;
    sBuffer.resize(1024);

    printf("training starts...\n");
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;
    for ( ; ; )
    {
        epoch.makeStep(loader, *pNetwork, lossComputer, lr);

        auto curTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> secondsSinceStart = curTime - startTime;
        double fMSecsPerTrainingStep = (secondsSinceStart.count() / lr.getNStepsMade()) * 1000;

        sprintf_s(&sBuffer[0], sBuffer.size() - 2,
            "nSteps: %d, avgPreError: %#.3g, avgPostError: %#.3g, avgLRate: %#.3g, "
            "MSecPerStep: %.2f, MB: %.2f\n",
            lr.getNStepsMade(), epoch.getAvgPreError(), epoch.getAvgPostError(),
            pNetwork->computeAvgLRStats(), fMSecsPerTrainingStep, (double)g_nCudaBytes / (1024 * 1024));
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
                pNetwork->serialize(writer);
            }
            lastSaveTime = curTime;
            printf("saving completed\n");
        }

        pNetwork->resetAvgLRStats();
    }
}

extern "C" __declspec(dllexport) OfflineTrainer * getOfflineTrainer()
{
    return new OfflineTrainer();
}