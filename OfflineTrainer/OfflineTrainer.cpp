#include <neural/atomsNetwork.h>

int main()
{
    NeuralTest::test();
    AtomsNetwork<float, 64> network;
    network.loadFromFile("1000steps_in.bin");

    FILE* fp = nullptr;
    fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "wt");
    fprintf(fp, "nSteps, error\n");
    fclose(fp);

    for ( ; ; )
    {
        network.trainAtomsNetwork(1024);

        printf("CompletedTrainSteps: %d, lastError: %e\n", network.getNCompletedTrainSteps(), network.getLastError());

        fp = nullptr;
        fopen_s(&fp, "C:\\atomNets\\offlineTrainer.csv", "a+");
        fprintf(fp, "%d, %e\n", network.getNCompletedTrainSteps(), network.getLastError());
        fclose(fp);
    }
}
