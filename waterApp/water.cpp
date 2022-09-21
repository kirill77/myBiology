#define NOMINMAX
#include "water.h"
#include "Easy3D/3rd_party/glfw/include/GLFW/glfw3.h"
#include "Easy3D/easy3d/renderer/text_renderer.h"
using TextRenderer = easy3d::TextRenderer;

template <class T>
void Water<T>::doTraining()
{
    NvU32 nStepsBefore = m_learningRates.getNStepsMade();
    m_neuralNetwork.initBatchForLastSimStep(m_batch);
    for ( ; ; )
    {
        m_fLastPreError = m_batch.makeMinimalProgress(m_neuralNetwork, m_lossComputer, m_learningRates);
        NvU32 nStepsAfter = m_learningRates.getNStepsMade();
        if (nStepsAfter > nStepsBefore + 500)
            break;
    }
}
template <class T>
void Water<T>::notifyKeyPress(int key, int modifiers)
{
    if (key == GLFW_KEY_EQUAL && (modifiers & GLFW_MOD_SHIFT))
    {
        m_fNNPropPrbb += 0.05f;
        m_fNNPropPrbb = std::min(m_fNNPropPrbb, 1.f);
    }
    if (key == GLFW_KEY_T)
    {
        m_doTraining = !m_doTraining;
    }
}
template <class T>
float Water<T>::drawText(TextRenderer* pTexter, float x, float y, float fDpiScaling, float fFontSize)
{
    char sBuffer[128];

    if (m_doTraining)
    {
        sprintf_s(sBuffer, "avgPreError: %#.3g", getAvgPreError());
        pTexter->draw(sBuffer,
            x * fDpiScaling, y * fDpiScaling, fFontSize, TextRenderer::ALIGN_LEFT, 1);
        y += 40;
    }
    if (m_fNNPropPrbbAccum > 0)
    {
        sprintf_s(sBuffer, "m_fNNPropPrbbAccum: %#.3g", m_fNNPropPrbbAccum);
        pTexter->draw(sBuffer,
            x * fDpiScaling, y * fDpiScaling, fFontSize, TextRenderer::ALIGN_LEFT, 1);
        y += 40;
    }

    return y;
}

//template struct Water<double>;
template struct Water<float>;
