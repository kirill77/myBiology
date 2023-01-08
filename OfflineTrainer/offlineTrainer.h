#pragma once

struct OfflineTrainer
{
    virtual int startTraining();
};

typedef OfflineTrainer* (*GET_OFFLINE_TRAINER_PROC)(); 
extern "C" __declspec(dllexport) OfflineTrainer *getOfflineTrainer();
