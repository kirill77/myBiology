#include <neural/atomsNetwork.h>
#include <chrono>

int main()
{
    NeuralTest::test();
    AtomsNetwork<float, 64> network;
    network.loadFromFile("1000steps_in.bin");

    FILE* fp = nullptr;
    fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "wt");
    fprintf(fp, "nSteps, fError, fLRate, MSecsPerStep\n");
    fclose(fp);

    auto start = std::chrono::high_resolution_clock::now();

    const NvU32 nStepsPerCycle = 1024;
    for (NvU32 nCycles = 0; ; ++nCycles)
    {
        network.trainAtomsNetwork(nStepsPerCycle);

        std::chrono::duration<double> elapsed_seconds = std::chrono::high_resolution_clock::now() - start;
        double fMSecsPerTrainingStep = (elapsed_seconds.count() / network.getNCompletedTrainSteps()) * 1000;

        fp = nullptr;
        fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "a+");
        fprintf(fp, "%d, %e, %e, %.2f\n", network.getNCompletedTrainSteps(), network.getLastError(), network.getLearningRate(), fMSecsPerTrainingStep);
        fclose(fp);
        printf("nSteps: %d, fError: %.2e, fLRate: %.2e, MSecsPerStep: %.2f\n", network.getNCompletedTrainSteps(), network.getLastError(), network.getLearningRate(), fMSecsPerTrainingStep);
    }
}
