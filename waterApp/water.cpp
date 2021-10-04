#include "water.h"

template <class _T>
void Water<_T>::splitRecursive(const NvU32 uNode, OcBoxStack<T>& stack)
{
    auto* pNode = &m_ocTree[uNode];
    nvAssert(pNode->isLeaf());
    pNode->m_nodeData.m_fTotalCharge.clear();
    if (pNode->getNPoints() <= 16) // no need to split further?
    {
        // update total node charge
        for (NvU32 uPoint = pNode->getFirstPoint(); uPoint < pNode->getEndPoint(); ++uPoint)
        {
            Atom& point = m_points[uPoint];
            pNode->m_nodeData.m_fTotalCharge += point.m_fCharge;
        }
        return;
    }
#if ASSERT_ONLY_CODE
    NvU32 dbgNPoints1 = pNode->getNPoints(), dbgNPoints2 = 0;
#endif
    pNode->split(stack, *this);
    NvU32 uFirstChild = m_ocTree[uNode].getFirstChild();
    for (NvU32 uChild = 0; uChild < 8; ++uChild)
    {
#if ASSERT_ONLY_CODE
        NvU32 dbgDepth = stack.getCurDepth();
        auto dbgBox = stack.getBox(dbgDepth);
#endif
        stack.push(uChild, uFirstChild + uChild);
#if ASSERT_ONLY_CODE
        nvAssert(stack.getCurDepth() == dbgDepth + 1);
        auto& node = m_ocTree[uFirstChild + uChild];
        dbgNPoints2 += node.getNPoints();
        if (stack.getCurDepth() == 1)
        {
            for (NvU32 u = node.getFirstPoint(); u < node.getEndPoint(); ++u)
            {
                nvAssert(stack.getBox(stack.getCurDepth()).includes(removeUnits(m_points[u].m_vPos)));
            }
        }
#endif
        splitRecursive(uFirstChild + uChild, stack);
        m_ocTree[uNode].m_nodeData.m_fTotalCharge += m_ocTree[uFirstChild + uChild].m_nodeData.m_fTotalCharge;
#if ASSERT_ONLY_CODE
        NvU32 childIndex = stack.pop();
        nvAssert(childIndex == uChild);
        nvAssert(stack.getCurDepth() == dbgDepth);
        nvAssert(stack.getBox(dbgDepth) == dbgBox);
#endif
    }
    nvAssert(dbgNPoints1 == dbgNPoints2);
}

template struct Water<double>;