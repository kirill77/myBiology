// neuralLauncher.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <stdio.h>
#include "Windows.h"
#include "OfflineTrainer/OfflineTrainer.h"

int main()
{
    HMODULE trainerModule = LoadLibrary(L"offlineTrainer.DLL");
    if (trainerModule == nullptr)
    {
        printf("Error: couldn't load offlineTrainer.DLL\n");
        return -1;
    }
    GET_OFFLINE_TRAINER_PROC getOfflineTrainer = (GET_OFFLINE_TRAINER_PROC)GetProcAddress(trainerModule, "getOfflineTrainer");
    if (!getOfflineTrainer)
    {
        printf("Error: couldn't get proc address for \"getOfflineTrainer\"\n");
        return -1;
    }
    printf("Loaded offline trainer successfully, start training\n");
    OfflineTrainer* pTrainer = getOfflineTrainer();
    pTrainer->startTraining();
}

