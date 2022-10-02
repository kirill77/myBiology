#define NOMINMAX
#include "Easy3D/easy3d/renderer/text_renderer.h"
using TextRenderer = easy3d::TextRenderer; 
#include "statusString.h"

void StatusString::print(const char* sFormat, ...)
{
	char sStackBuffer[256];
	std::vector<char> pHeapBuffer;
	char *sTmp = sStackBuffer;
	size_t curTmpSize = sizeof(sStackBuffer) - 2;

	// keep trying to print stuff until we have enough memory to do it
	for ( ; ; )
	{
		va_list args;
		va_start(args, sFormat);
		int result = vsnprintf_s(sTmp, curTmpSize, _TRUNCATE, sFormat, args);
		va_end(args);
		if (result >= 0)
		{
			break;
		}
		pHeapBuffer.clear();
		pHeapBuffer.resize(curTmpSize * 2);
		sTmp = &pHeapBuffer[0];
		curTmpSize = pHeapBuffer.size() - 2;
	}

	set(sTmp);
}
void StatusString::set(const char* s)
{
    m_sLastStatus = s;
    m_lastStatusTS = std::chrono::system_clock::now();
}
// returns new y coordinate (depends on whether something was printed or not)
float StatusString::drawText(class easy3d::TextRenderer* pTexter, float x, float y, float fDpiScaling, float fFontSize)
{
    if (m_sLastStatus.size() == 0)
        return y;

    auto curTime = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = curTime - m_lastStatusTS;
    if (seconds.count() > 10)
    {
        m_sLastStatus.clear();
        return y;
    }

    pTexter->draw(m_sLastStatus.c_str(),
        x * fDpiScaling, y * fDpiScaling, fFontSize, TextRenderer::ALIGN_LEFT, 1);
    y += 40;

    return y;
}
