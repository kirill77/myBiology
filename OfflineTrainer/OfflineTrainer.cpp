#include <mycuda1/atomsNetwork.h>
#include <chrono>
#include <filesystem>

int main()
{
    NeuralTest::test();
    AtomsNetwork<float, 64> network;

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
        if (!path.empty())
        {
            printf("loading %S\n", path.c_str());
            network.loadFromFile(path);
        }
        else
        {
            printf("loading %s\n", "networkFromWaterApp.bin");
            network.loadFromFile(L"networkFromWaterApp.bin");
        }
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastSaveTime = startTime;

    const NvU32 nLoadedTrainSteps = network.getNCompletedTrainSteps();
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
        network.trainAtomsNetwork(nStepsPerCycle);

        auto curTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> secondsInTraining = curTime - startTime;
        NvU32 nTrainStepsMadeThisSession = network.getNCompletedTrainSteps() - nLoadedTrainSteps;
        double fMSecsPerTrainingStep = (secondsInTraining.count() / nTrainStepsMadeThisSession) * 1000;

        {
            FILE* fp = nullptr;
            fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "a+");
            if (fp != nullptr)
            {
                fprintf(fp, "%d,  %#.3g,  %#.3g, %.2f\n", network.getNCompletedTrainSteps(), network.getLastError(), network.getFilteredLearningRate(), fMSecsPerTrainingStep);
                fclose(fp);
            }
        }

        std::chrono::duration<double> secondsSinceLastSave = curTime - lastSaveTime;
        if (secondsSinceLastSave.count() > 60)
        {
            printf("saving nSteps=%d...\n", network.getNCompletedTrainSteps());
            wchar_t sBuffer[32];
            swprintf_s(sBuffer, L"trained_%d.bin", network.getNCompletedTrainSteps());
            network.saveToFile(sBuffer);
            lastSaveTime = curTime;
            printf("saving completed\n");
        }

        printf("nSteps: %d, fError: %#.3g, fLRate: %#.3g, MSecsPerStep: %.2f\n", network.getNCompletedTrainSteps(), network.getLastError(), network.getFilteredLearningRate(), fMSecsPerTrainingStep);
    }
}
