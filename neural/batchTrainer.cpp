#include "batchTrainer.h"
#include "network.h"

void BatchTrainer::init(NeuralNetwork &network, std::vector<TensorRef> inputs, std::vector<TensorRef> wantedOutputs)
{
    (*this) = BatchTrainer();

    m_inputs = inputs;
    m_wantedOutputs = wantedOutputs;
    m_loss.init(m_wantedOutputs[0]->getDims());

    m_pLayerOutputs.resize(network.getNLayers());
    for (NvU32 u = 0; u < m_pLayerOutputs.size(); ++u)
    {
        network.getLayer(u).allocateBatchData(m_pLayerOutputs[u], n());
    }

    m_lr.init(network.getNLayers());
}
void BatchTrainer::makeMinimalProgress(NeuralNetwork& network, LossComputer &lossComputer)
{
    network.forwardPass(*this);
    float fError = 0;
    computeLoss(lossComputer, &fError);
    m_lr.setInitialError(fError);
    network.saveCurrentStateToBackup();

    for (NvU32 u = 0; u < m_lr.getNStepsToMake(); ++u)
    {
        backwardPass(network, lossComputer);
        network.forwardPass(*this);
    }

    float fCurrentError = 0;
    computeLoss(lossComputer, &fCurrentError);
    bool bShouldRedo = true;
    m_lr.notifyNewError(fCurrentError, bShouldRedo);
    if (bShouldRedo)
    {
        network.restoreStateFromBackup();
        network.forwardPass(*this);
    }
    else
    {
        network.saveCurrentStateToBackup();
    }
}
NvU32 LearningRates::notifyNewError(float fError, bool& bShouldRedo)
{
    bool bLocalIncreaseOnPrevStep = m_bLocalIncreaseOnPrevStep;
    m_bLocalIncreaseOnPrevStep = false;
    m_nStepsMade += m_nStepsToMake;
    bool bErrorHasImproved = (isfinite(fError) && fError < m_fPrevError);
    bShouldRedo = !bErrorHasImproved;
    for (; ; ) // loop until the decision is made
    {
        if (bErrorHasImproved)
        {
            if (m_isGlobal)
            {
                if (m_nStepsToMake < 128)
                {
                    m_nStepsToMake *= 2;
                    break;
                }
            }
            m_isGlobal = false;
            if (bLocalIncreaseOnPrevStep)
            {
                RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                float fNewErrorDecreaseRate = (m_fPrevError - fError) / m_nStepsToMake;
                // did learning get slower?
                if (fNewErrorDecreaseRate < m_fPrevErrorDecreaseRate)
                {
                    nvAssert(r.m_fPrevValue < r.m_fValue);
                    r.m_fValue = r.m_fPrevValue;
                    r.m_uAttemptThreshold = std::min(1024u, std::max(2u, (NvU32)(r.m_uAttemptThreshold * 1.5)));
                    break;
                }
                // if the attempt was good - reward by decrementing the threshold
                r.m_uAttemptThreshold = std::max(1u, r.m_uAttemptThreshold - 1);
            }
            // loop some limited amount trying to find which learning rate to increase
            for (NvU32 u = std::min((NvU32)m_pRatesInfo.size(), 8u); u != 0; --u)
            {
                m_uLastAttemptedRate = (m_uLastAttemptedRate + 1) % m_pRatesInfo.size();
                RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                if (++r.m_uAttemptCounter >= r.m_uAttemptThreshold)
                {
                    m_fPrevErrorDecreaseRate = (m_fPrevError - fError) / m_nStepsToMake;
                    r.m_uAttemptCounter = 0;
                    r.m_fPrevValue = r.m_fValue;
                    r.m_fValue *= 2;
                    m_bLocalIncreaseOnPrevStep = true;
                    break;
                }
            }
            break;
        }
        else
        {
            if (bLocalIncreaseOnPrevStep)
            {
                RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                nvAssert(r.m_fPrevValue < r.m_fValue);
                r.m_fValue = r.m_fPrevValue;
                r.m_uAttemptThreshold = std::min(1024u, std::max(2u, r.m_uAttemptThreshold * 2));
                break;
            }
            // if it wasn't global before - make rate increases less likely
            if (!m_isGlobal)
            {
                for (NvU32 u = 0; u < m_pRatesInfo.size(); ++u)
                {
                    RateInfo& r = m_pRatesInfo[m_uLastAttemptedRate];
                    r.m_uAttemptThreshold = std::min(1024u, std::max(2u, r.m_uAttemptThreshold * 2));
                }
            }
            m_isGlobal = true;
            m_nStepsToMake = 1;
            for (NvU32 u = 0; u < m_pRatesInfo.size(); ++u)
            {
                m_pRatesInfo[u].m_fValue /= 2;
            }
            break;
        }
    }
    if (bErrorHasImproved)
    {
        m_fPrevError = fError;
    }
    return m_nStepsToMake;
}
void BatchTrainer::computeLoss(LossComputer& lossComputer, float *pErrorPtr)
{
    const std::vector<TensorRef>& outputs = m_pLayerOutputs.rbegin()->m_outputs;
    nvAssert(outputs.size() == 1 && m_wantedOutputs.size() == 1);
    Tensor<float>& output = (*outputs[0]);
    Tensor<float>& wantedOutput = (*m_wantedOutputs[0]);
    lossComputer.compute(output, wantedOutput, m_loss, pErrorPtr);
}
void BatchTrainer::backwardPass(NeuralNetwork& network, LossComputer& lossComputer)
{
    NvU32 nLayers = network.getNLayers();
    NvU32 uLayer = nLayers - 1;
    while (uLayer < nLayers)
    {
        std::vector<TensorRef>& inputs = getInputs(uLayer);

        // we don't need to compute deltaInputs for the layer 0
        std::vector<TensorRef>* pDeltaInputs = (uLayer == 0) ? nullptr : &accessLayerData(uLayer - 1).m_deltaOutputs;
        float fBiasesLR = m_lr.getLearningRate(uLayer);
        float fWeightsLR = m_lr.getLearningRate(uLayer);
        m_fLRSum += fBiasesLR + fWeightsLR;
        m_nLRSamples += 2;
        if (uLayer == nLayers - 1)
        {
            computeLoss(lossComputer);
            network.getLayer(uLayer).backward(inputs, m_loss,
                fBiasesLR, fWeightsLR, *this, pDeltaInputs);
        }
        else
        {
            network.getLayer(uLayer).backward(inputs,
                *accessLayerData(uLayer).m_deltaOutputs[0], fBiasesLR, fWeightsLR,
                *this, pDeltaInputs);
        }
        --uLayer;
    }
}

