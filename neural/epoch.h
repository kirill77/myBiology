#pragma once

#include <vector>
#include "neural/batchTrainer.h"

struct Epoch
{
   void init(struct NeuralNetwork &network);

   void makeStep(NeuralNetwork &network, struct LossComputer& lossComputer, struct LearningRates &lr);

   float getAvgPreError() const { return m_fAvgPreError; }
   float getAvgPostError() const { return m_fAvgPostError; }

private:
   std::vector<BatchTrainer> m_batches;
   float m_fAvgPreError = 0, m_fAvgPostError = 0;
};