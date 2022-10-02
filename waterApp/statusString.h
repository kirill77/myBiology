#pragma once

#include <string>
#include <chrono>

namespace easy3d {
    class TextRenderer;
};

struct StatusString
{
    void print(const char *sFormat, ...);
    void set(const char* s);
    // returns new y coordinate (depends on whether something was printed or not)
    float drawText(class easy3d::TextRenderer* pTexter, float x, float y, float fDpiScaling, float fFontSize);

private:
    // for reporting status
    std::string m_sLastStatus;
    std::chrono::system_clock::time_point m_lastStatusTS;
};