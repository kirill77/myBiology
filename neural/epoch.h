#pragma once

#include <vector>
#include "neural/batch.h"

struct Epoch
{
   void makeStep(NeuralNetwork &network, struct LossComputer& lossComputer, struct LearningRates &lr);

   float getAvgPreError() const { return m_fAvgPreError; }
   float getAvgPostError() const { return m_fAvgPostError; }

private:
   float m_fAvgPreError = 0, m_fAvgPostError = 0;
};