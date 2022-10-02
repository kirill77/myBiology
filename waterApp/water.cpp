#include "Easy3D/3rd_party/glfw/include/GLFW/glfw3.h"
#include "water.h"

template <class T>
void Water<T>::notifyKeyPress(int key, int modifiers)
{
    if (key == GLFW_KEY_S && (modifiers & GLFW_MOD_CONTROL))
    {
        char sBuffer[128];
        sprintf_s(sBuffer, "c:\\atomNets\\water_%d.bin", m_neuralNetwork.getNStoredSimSteps());
        MyWriter writer(sBuffer);
        m_neuralNetwork.serialize(writer);
        m_statusString.print("%s saved", sBuffer);
    }
    if (key == GLFW_KEY_R)
    {
        this->m_c.setRandomization(!this->m_c.getRandomization());
        m_statusString.print("randomization of forces: %s", this->m_c.getRandomization() ? "enabled" : "disabled");
    }
}

template <class T>
float Water<T>::drawText(easy3d::TextRenderer* pTexter, float x, float y, float fDpiScaling, float fFontSize)
{
    return m_statusString.drawText(pTexter, x, y, fDpiScaling, fFontSize);
}

//template struct Water<double>;
template struct Water<float>;
