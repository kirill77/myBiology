#pragma once

#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGUniform.h"
#include "MonteCarlo/distributions.h"

template <class T>
struct Neuron
{
    typedef T FLOAT_TYPE;
    Neuron() : m_rng((NvU32)time(nullptr))
    {
        m_bbox.m_vMin = makevector<T, 3>(-m_fRadius);
        m_bbox.m_vMax = makevector<T, 3>( m_fRadius);
        // create IONS
        for (NvU32 u = 0; u < 1024; ++u)
        {
            Point point;
            if ((u % 100) < 50) // percentage of K ions
            {
                point.m_flags |= FLAG_K_ION;
            }
            else
            {
                point.m_flags |= FLAG_NA_ION;
            }
            for (NvU32 u = 0; u < 3; ++u)
            {
                point.m_pos[u] = m_rng.generate01();
            }
            // generates points inside unit (-1,1) sphere and then multiplies by radius
            point.m_pos = SphereVolumeDistribution<T>::generate(point.m_pos) * m_fRadius;
            nvAssert(m_bbox.includes(point.m_pos));
            m_points.push_back(point);
        }

        // create root oc-tree node
        m_ocTree.resize(1);
        m_ocTree[0].initLeaf(0, (NvU32)m_points.size());
        // initialize stack
        OcTreeBoxStack<T, 32> stack;
        stack.init(m_bbox);
        // split oc-tree recursively to get small number of points per leaf
        split(0, stack);
    }

    enum FLAGS { FLAG_ION_PUMP = 1, FLAG_K_ION = 2, FLAG_NA_ION = 4 };
    struct Point
    {
        NvU32 m_flags = 0;
        rtvector<T, 3> m_pos;
    };
    inline std::vector<Point>& points()
    {
        return m_points;
    }

    void makeTimeStep(double fDeltaT_MS)
    {
        for (NvU32 u = 0; u < m_points.size(); ++u)
        {
            m_points[u].m_pos[0] += (T)((m_rng.generate01() * 2 - 1) * 0.01);
            m_points[u].m_pos[1] += (T)((m_rng.generate01() * 2 - 1) * 0.01);
            m_points[u].m_pos[2] += (T)((m_rng.generate01() * 2 - 1) * 0.01);
        }
    }
    NvU32 getNNodes() const { return (NvU32)m_ocTree.size(); }
    OcTreeNode<Neuron>& accessNode(NvU32 index) { return m_ocTree[index]; }
    void resizeNodes(NvU32 nNodes) { m_ocTree.resize(nNodes); }
    const rtvector<T, 3>& getPoint(NvU32 u) const { return m_points[u].m_pos; }
    void swapPoints(NvU32 u1, NvU32 u2) { nvSwap(m_points[u1], m_points[u2]); }

private:
    BBox3<T> m_bbox;
    void split(NvU32 uNode, OcTreeBoxStack<T, 32> &stack)
    {
        auto* pNode = &m_ocTree[uNode];
        nvAssert(pNode->isLeaf());
        if (pNode->getNPoints() > 16)
        {
#if ASSERT_ONLY_CODE
            NvU32 dbgNPoints1 = pNode->getNPoints(), dbgNPoints2 = 0;
#endif
            pNode->split(stack.getCurBox().computeCenter(), *this);
            NvU32 uFirstChild = m_ocTree[uNode].getFirstChild();
            for (NvU32 u = 0; u < 8; ++u)
            {
#if ASSERT_ONLY_CODE
                NvU32 dbgDepth = stack.getCurDepth();
                auto dbgBox = stack.getCurBox();
#endif
                stack.push(u);
#if ASSERT_ONLY_CODE
                nvAssert(stack.getCurDepth() == dbgDepth + 1);
                auto& node = m_ocTree[uFirstChild + u];
                dbgNPoints2 += node.getNPoints();
                if (stack.getCurDepth() == 1)
                {
                    for (NvU32 u = node.getFirstPoint(); u < node.getEndPoint(); ++u)
                    {
                        nvAssert(stack.getCurBox().includes(m_points[u].m_pos));
                    }
                }
#endif
                split(uFirstChild + u, stack);
                NvU32 childIndex = stack.pop();
                nvAssert(childIndex == u);
                nvAssert(stack.getCurBox() == dbgBox);
                nvAssert(stack.getCurDepth() == dbgDepth);
            }
            nvAssert(dbgNPoints1 == dbgNPoints2);
        }
    }
    T m_fRadius = MyNumeric<T>::microMeter() * 100; // it's a circle for simplicity
    std::vector<Point> m_points;
    std::vector<OcTreeNode<Neuron>> m_ocTree;
    RNGUniform m_rng;
};