#pragma once

#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGUniform.h"
#include "MonteCarlo/distributions.h"

template <class T>
struct Neuron
{
    typedef T FLOAT_TYPE;
    struct NODE_DATA // data that we store in each node
    {
        int m_iTotalCharge; // can be positive or negative
        rtvector<T, 3> m_vForce;
    };

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
                point.initAs(FLAG_K_ION);
            }
            else
            {
                point.initAs(FLAG_NA_ION);
            }
            for (NvU32 u = 0; u < 3; ++u)
            {
                point.m_vPos[u] = m_rng.generate01();
            }
            // generates points inside unit (-1,1) sphere and then multiplies by radius
            point.m_vPos = SphereVolumeDistribution<T>::generate(point.m_vPos) * m_fRadius;
            nvAssert(m_bbox.includes(point.m_vPos));
            m_points.push_back(point);
        }

        // create root oc-tree node
        m_ocTree.resize(1);
        m_ocTree[0].initLeaf(0, (NvU32)m_points.size());
        // initialize stack
        OcBoxStack<T> stack;
        stack.init(m_bbox);
        // split oc-tree recursively to get small number of points per leaf
        splitRecursive(0, stack);
    }

    enum FLAGS { FLAG_ION_PUMP = 1, FLAG_K_ION = 2, FLAG_NA_ION = 4 };
    struct Point
    {
        void initAs(FLAGS flags)
        {
            m_flags = flags;
            switch (flags)
            {
            case FLAG_NA_ION:
            case FLAG_K_ION:
                m_iCharge = 1;
                break;
            default:
                nvAssert(false);
            }
        }
        NvU32 m_flags = 0;
        int m_iCharge = 0;
        rtvector<T, 3> m_vPos, m_vForce;
    };
    inline std::vector<Point>& points()
    {
        return m_points;
    }

    void makeTimeStep(double fDeltaT_MS)
    {
        initForcesRecursive(0);
        m_ocTree[0].computeForces(0, m_bbox, *this);
        for (NvU32 u = 0; u < m_points.size(); ++u)
        {
            m_points[u].m_vPos[0] += (T)((m_rng.generate01() * 2 - 1) * 0.01 * m_fRadius);
            m_points[u].m_vPos[1] += (T)((m_rng.generate01() * 2 - 1) * 0.01 * m_fRadius);
            m_points[u].m_vPos[2] += (T)((m_rng.generate01() * 2 - 1) * 0.01 * m_fRadius);
        }
    }
    NvU32 getNNodes() const { return (NvU32)m_ocTree.size(); }
    OcTreeNode<Neuron>& accessNode(NvU32 index) { return m_ocTree[index]; }
    void resizeNodes(NvU32 nNodes) { m_ocTree.resize(nNodes); }
    const rtvector<T, 3>& getPoint(NvU32 u) const { return m_points[u].m_vPos; }
    void swapPoints(NvU32 u1, NvU32 u2) { nvSwap(m_points[u1], m_points[u2]); }
    static T accuracyThreshold() { return (T)0.4; } // smaller -> more accurate. 0 means absolutely accurate O(N^2) algorithm

    // returns true if contributions between those two boxes are fully accounted for (either just now or before - at higher level of hierarchy)
    bool addNode2NodeContributions(NvU32 node1Index, const OcBoxStack<T>& stack1, NvU32 node2Index, const OcBoxStack<T>& stack2)
    {
        // check accuracy at minimum common depth
        NvU32 uMinDepth = std::min(stack1.getCurDepth(), stack2.getCurDepth());
        auto& bbox1 = stack1.getBox(uMinDepth);
        auto& bbox2 = stack2.getBox(uMinDepth);
        auto vDir = bbox2.computeCenter() - bbox1.computeCenter();
        T fDistSqr = lengthSquared(vDir);
        T fDist = sqrt(fDistSqr);
        // bbox1 and bbox2 must be cubes of the same size, but to make algorithm non-sensitive to swapping of boxes - take their sum
        T fSize = (bbox1.m_vMax[0] - bbox1.m_vMin[0]) + (bbox2.m_vMax[0] - bbox2.m_vMin[0]);

        if (fSize / fDist > accuracyThreshold())
            return false;

        // if depth is not the same, this interaction must have been taken into account higher up the stack
        if (stack1.getCurDepth() == stack2.getCurDepth())
        {
            auto& node1 = m_ocTree[node1Index];
            auto& node2 = m_ocTree[node2Index];
            T fForce = (node2.m_nodeData.m_iTotalCharge * node2.m_nodeData.m_iTotalCharge) / fDistSqr;
            auto vForce = vDir * (fForce / fDist);
            node1.m_nodeData.m_vForce += vForce;
            node2.m_nodeData.m_vForce -= vForce;
        }

        return true;
    }
    bool addNode2PointContributions(NvU32 nodeIndex, const OcBoxStack<T>& stack, NvU32 pointIndex)
    {
        auto& bbox = stack.getBox(stack.getCurDepth());
        auto& point = m_points[pointIndex];
        auto vDir = point.m_vPos - bbox.computeCenter();
        T fDistSqr = lengthSquared(vDir);
        T fDist = sqrt(fDistSqr);
        T fSize = (bbox.m_vMax[0] - bbox.m_vMin[0]);

        if (fSize / fDist > accuracyThreshold())
            return false;

        auto& node = m_ocTree[nodeIndex];
        T fForce = (node.m_nodeData.m_iTotalCharge * point.m_iCharge) / fDistSqr;
        auto vForce = vDir * (fForce / fDist);
        node.m_nodeData.m_vForce += vForce;
        point.m_vForce -= vForce;

        return true;
    }
    void addPoint2PointContributions(NvU32 point1Index, NvU32 point2Index)
    {
        auto& point1 = m_points[point1Index];
        auto& point2 = m_points[point2Index];
        auto vDir = point2.m_vPos - point1.m_vPos;
        T fDistSqr = lengthSquared(vDir);
        T fDist = sqrt(fDistSqr);

        T fForce = (point1.m_iCharge * point2.m_iCharge) / fDistSqr;
        auto vForce = vDir * (fForce / fDist);
        point1.m_vForce += vForce;
        point2.m_vForce -= vForce;
    }

private:
    void initForcesRecursive(NvU32 uNode)
    {
        auto& node = m_ocTree[uNode];
        node.m_nodeData.m_vForce = rtvector<T, 3>({ 0, 0, 0 });
        node.m_nodeData.m_iTotalCharge = 0;
        if (node.isLeaf())
        {
            for (NvU32 uPoint = node.getFirstPoint(); uPoint < node.getEndPoint(); ++uPoint)
            {
                auto& point = m_points[uPoint];
                point.m_vForce = rtvector<T, 3>({ 0, 0, 0 });
                node.m_nodeData.m_iTotalCharge += point.m_iCharge;
            }
            return;
        }
        NvU32 uFirstChild = m_ocTree[uNode].getFirstChild();
        for (NvU32 uChild = 0; uChild < 8; ++uChild)
        {
            initForcesRecursive(uFirstChild + uChild);
            node.m_nodeData.m_iTotalCharge += m_ocTree[uFirstChild + uChild].m_nodeData.m_iTotalCharge;
        }
    }
    BBox3<T> m_bbox;
    void splitRecursive(NvU32 uNode, OcBoxStack<T> &stack)
    {
        auto* pNode = &m_ocTree[uNode];
        nvAssert(pNode->isLeaf());
        if (pNode->getNPoints() > 16)
        {
#if ASSERT_ONLY_CODE
            NvU32 dbgNPoints1 = pNode->getNPoints(), dbgNPoints2 = 0;
#endif
            pNode->split(stack.getBox(stack.getCurDepth()).computeCenter(), *this);
            NvU32 uFirstChild = m_ocTree[uNode].getFirstChild();
            for (NvU32 u = 0; u < 8; ++u)
            {
#if ASSERT_ONLY_CODE
                NvU32 dbgDepth = stack.getCurDepth();
                auto dbgBox = stack.getBox(dbgDepth);
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
                        nvAssert(stack.getBox(stack.getCurDepth()).includes(m_points[u].m_vPos));
                    }
                }
#endif
                splitRecursive(uFirstChild + u, stack);
                NvU32 childIndex = stack.pop();
                nvAssert(childIndex == u);
                nvAssert(stack.getCurDepth() == dbgDepth);
                nvAssert(stack.getBox(dbgDepth) == dbgBox);
            }
            nvAssert(dbgNPoints1 == dbgNPoints2);
        }
    }
    T m_fRadius = MyNumeric<T>::microMeter() * 100; // it's a circle for simplicity
    std::vector<Point> m_points;
    std::vector<OcTreeNode<Neuron>> m_ocTree;
    RNGUniform m_rng;
};