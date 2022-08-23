#include <neural/atomsNetwork.h>
#include <chrono>
#include <filesystem>

int main()
{
    NeuralTest::test();

    AtomsNetwork<float, 64> network;
    BatchTrainer batchTrainer;

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
            path = L"c:\\atomNets\\networkFromWaterApp.bin";
        }
        printf("loading %S\n", path.c_str());
        MyReader reader(path);
        network.serialize(reader);
        batchTrainer.serialize(reader);
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;

    const NvU32 nLoadedTrainSteps = batchTrainer.getNStepsMade();
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

    const NvU32 nStepsPerCycle = 1024;
    for (NvU32 nCycles = 0; ; ++nCycles)
    {
        for ( ; ; )
        {
            batchTrainer.makeMinimalProgress(network);
            if (batchTrainer.getNStepsMade() >= (nCycles + 1) * nStepsPerCycle)
                break;
        }

        auto curTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> secondsInTraining = curTime - startTime;
        NvU32 nTrainStepsMadeThisSession = batchTrainer.getNStepsMade() - nLoadedTrainSteps;
        double fMSecsPerTrainingStep = (secondsInTraining.count() / nTrainStepsMadeThisSession) * 1000;

        {
            FILE* fp = nullptr;
            fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "a+");
            if (fp != nullptr)
            {
                fprintf(fp, "%d,  %#.3g,  %#.3g, %.2f\n", batchTrainer.getNStepsMade(), batchTrainer.getLastError(), network.getFilteredLearningRate(), fMSecsPerTrainingStep);
                fclose(fp);
            }
        }

        std::chrono::duration<double> secondsSinceLastSave = curTime - lastSaveTime;
        if (secondsSinceLastSave.count() > 60)
        {
            printf("saving nSteps=%d...\n", batchTrainer.getNStepsMade());
            wchar_t sBuffer[32];
            swprintf_s(sBuffer, L"c:\\atomNets\\trained_%d.bin", batchTrainer.getNStepsMade());
            {
            MyWriter writer(sBuffer);
            network.serialize(writer);
            batchTrainer.serialize(writer);
            }
            lastSaveTime = curTime;
            printf("saving completed\n");
        }

        printf("nSteps: %d, fError: %#.3g, fLRate: %#.3g, MSecsPerStep: %.2f\n", batchTrainer.getNStepsMade(), batchTrainer.getLastError(), network.getFilteredLearningRate(), fMSecsPerTrainingStep);
    }
}
