#include "water.h"

template <class _T>
void Water<_T>::splitRecursive(OcBoxStack<T>& curStack)
{
    NvU32 parentNodeIndex = curStack.getCurNodeIndex();
    auto* pNode = &m_ocTree[parentNodeIndex];
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
    pNode->split(curStack, *this);
    NvU32 uFirstChild = m_ocTree[parentNodeIndex].getFirstChild();
    for (NvU32 uChild = 0; uChild < 8; ++uChild)
    {
#if ASSERT_ONLY_CODE
        NvU32 dbgDepth = curStack.getCurDepth();
        auto dbgBox = curStack.getBox(dbgDepth);
#endif
        NvU32 childNodeIndex = uFirstChild + uChild;
        curStack.push(uChild, childNodeIndex);
#if ASSERT_ONLY_CODE
        nvAssert(curStack.getCurDepth() == dbgDepth + 1);
        auto& node = m_ocTree[childNodeIndex];
        dbgNPoints2 += node.getNPoints();
        if (curStack.getCurDepth() == 1)
        {
            for (NvU32 u = node.getFirstPoint(); u < node.getEndPoint(); ++u)
            {
                nvAssert(curStack.getBox(curStack.getCurDepth()).includes(removeUnits(m_points[u].m_vPos)));
            }
        }
#endif
        splitRecursive(curStack);
        m_ocTree[parentNodeIndex].m_nodeData.m_fTotalCharge += m_ocTree[childNodeIndex].m_nodeData.m_fTotalCharge;
        NvU32 childIndex = curStack.pop();
#if ASSERT_ONLY_CODE
        // check that after we pop() we get the same box we had before push()
        nvAssert(childIndex == uChild);
        nvAssert(curStack.getCurDepth() == dbgDepth);
        nvAssert(curStack.getBox(dbgDepth) == dbgBox);
#endif
    }
    nvAssert(dbgNPoints1 == dbgNPoints2);
}

template struct Water<double>;