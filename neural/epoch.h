#pragma once

#include <vector>
#include "neural/batchTrainer.h"

struct Epoch
{
   void init(struct NeuralNetwork &network);

   void makeStep(NeuralNetwork &network, struct LossComputer& lossComputer, struct LearningRates &lr);

   float getAvgError() const { return m_fAvgError; }

private:
   std::vector<BatchTrainer> m_batches;
   float m_fAvgError = 0;
};