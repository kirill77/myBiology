#include "water.h"

template <class _T>
void Water<_T>::splitRecursive(OcBoxStack<T>& curStack)
{
    NvU32 parentNodeIndex = curStack.getCurNodeIndex();
    auto* pNode = &m_ocTree.m_nodes[parentNodeIndex];
    nvAssert(pNode->isLeaf());
    if (pNode->getNPoints() <= 16) // no need to split further?
    {
        return;
    }
#if ASSERT_ONLY_CODE
    NvU32 dbgNPoints1 = pNode->getNPoints(), dbgNPoints2 = 0;
#endif
    m_ocTree.split(curStack);
    NvU32 uFirstChild = m_ocTree.m_nodes[parentNodeIndex].getFirstChild();
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
        auto& node = m_ocTree.m_nodes[childNodeIndex];
        dbgNPoints2 += node.getNPoints();
        if (curStack.getCurDepth() == 1)
        {
            for (NvU32 u = node.getFirstPoint(); u < node.getEndPoint(); ++u)
            {
                nvAssert(curStack.getBox(curStack.getCurDepth()).includes(removeUnits(m_points[u].m_vPos[0])));
            }
        }
#endif
        splitRecursive(curStack);
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