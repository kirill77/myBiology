#pragma once

#include "basics/myunits.h"
#include "ocTree/ocTree.h"
#include "MonteCarlo/RNGUniform.h"
#include "MonteCarlo/distributions.h"

template <class _T>
struct Neuron
{
    typedef _T T;

    struct NODE_DATA // data that we store in each node
    {
        MyUnits<int> m_iTotalCharge; // can be positive or negative
    };

    Neuron() : m_rng((NvU32)time(nullptr))
    {
        m_bbox.m_vMin = makeVector<MyUnits<T>, 3>(-m_fRadius);
        m_bbox.m_vMax = makeVector<MyUnits<T>, 3>( m_fRadius);
        // create IONS
        for (NvU32 u = 0; u < 1024; ++u)
        {
            Point point;
            if ((u % 100) < 50) // percentage of K ions
            {
                point.initAs(FLAG_K_ION);
                point.m_fMass = MyUnits<T>::dalton() * (T)39.09831;
            }
            else
            {
                point.initAs(FLAG_NA_ION);
                point.m_fMass = MyUnits<T>::dalton() * (T)22.989769282;
            }
            // generates points inside unit (-1,1) sphere and then multiplies by radius
            rtvector<T, 3> vPosInsideUnitSphere = SphereVolumeDistribution<T>::generate(rtvector<T, 3>({ m_rng.generate01(), m_rng.generate01(), m_rng.generate01() }));
            point.m_vPos = setUnits<MyUnits<T>>(vPosInsideUnitSphere) * m_fRadius;
            nvAssert(m_bbox.includes(point.m_vPos));
            m_points.push_back(point);
        }

        // create root oc-tree node
        m_ocTree.resize(1);
        m_ocTree[0].initLeaf(0, (NvU32)m_points.size());
        // initialize stack
        OcBoxStack<T> stack(0, removeUnits(m_bbox));
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
                m_iCharge = MyUnits<int>::electron();
                break;
            default:
                nvAssert(false);
            }
        }
        NvU32 m_flags = 0;
        MyUnits<int> m_iCharge;
        rtvector<MyUnits<T>,3> m_vPos, m_vSpeed, m_vForce;
        MyUnits<T> m_fMass;
    };
    inline std::vector<Point>& points()
    {
        return m_points;
    }

    void makeTimeStep(MyUnits<T> fDeltaT)
    {
        initForcesRecursive(0);
#if ASSERT_ONLY_CODE
        m_dbgNContributions = 0;
#if RANDOM_POINT_DEBUG
        m_dbgUDstPoint = m_rng.generateUnsigned(0, (NvU32)m_points.size());
        m_dbgSrcAccounted.resize(m_points.size());
        for (NvU32 u = 0; u < m_dbgSrcAccounted.size(); ++u)
        {
            m_dbgSrcAccounted[u] = 0;
        }
#endif
#endif
        m_ocTree[0].computeForces(0, removeUnits(m_bbox), *this);
#if ASSERT_ONLY_CODE
        nvAssert(m_dbgNContributions == m_points.size() * (m_points.size() - 1));
#if RANDOM_POINT_DEBUG
        for (NvU32 u = 0; u < m_dbgSrcAccounted.size(); ++u)
        {
            nvAssert((m_dbgSrcAccounted[u] == 1) == (u != m_dbgUDstPoint));
        }
#endif
#endif
        // change speed of each point according to force and advect points according to new speed
        for (NvU32 uPoint = 0; uPoint < m_points.size(); ++uPoint)
        {
            auto& point = m_points[uPoint];
            rtvector<MyUnits<T>,3> vAcceleration = newtonLaw(point.m_fMass, point.m_vForce);
            point.m_vSpeed += vAcceleration * fDeltaT;
            point.m_vPos += point.m_vSpeed * fDeltaT;

            // ions can't fly farther than membrane allows
            auto dist = length(point.m_vPos);
            if (dist > m_fRadius)
            {
                point.m_vPos *= (m_fRadius / dist);
            }
        }
    }
    NvU32 getNNodes() const { return (NvU32)m_ocTree.size(); }
    OcTreeNode<Neuron>& accessNode(NvU32 index) { return m_ocTree[index]; }
    void resizeNodes(NvU32 nNodes) { m_ocTree.resize(nNodes); }
    const rtvector<T, 3>& getPoint(NvU32 u) const { return removeUnits(m_points[u].m_vPos); }
    void swapPoints(NvU32 u1, NvU32 u2) { nvSwap(m_points[u1], m_points[u2]); }
    static T accuracyThreshold() { return (T)0.4; } // smaller -> more accurate. 0 means absolutely accurate O(N^2) algorithm

    // returns true if contributions between those two boxes are fully accounted for (either just now or before - at higher level of hierarchy)
    bool addNode2LeafContribution(NvU32 dstLeafIndex, const OcBoxStack<T>& dstLeafStack, NvU32 srcNodeIndex, const OcBoxStack<T>& srcNodeStack)
    {
        nvAssert(m_ocTree[dstLeafIndex].getNPoints() && m_ocTree[srcNodeIndex].getNPoints());
        // check if we can treat srcNode as one point as opposed to looking at its individual sub-boxes or points
        if (!dstLeafStack.isDescendent(srcNodeStack))
        {
            const auto& dstBox = setUnits<MyUnits<T>>(dstLeafStack.getCurBox());
            const auto& srcBox = setUnits<MyUnits<T>>(srcNodeStack.getCurBox());
            rtvector<MyUnits<T>,3> dstBoxCenter = dstBox.computeCenter();
            rtvector<MyUnits<T>,3> srcBoxCenter = srcBox.computeCenter();
            rtvector<MyUnits<T>,3> vDir = dstBoxCenter - srcBoxCenter;
            MyUnits<T> fDistSqr = lengthSquared(vDir);
            MyUnits<T> fDist = fDistSqr.sqrt();
            MyUnits<T> dstBoxSize = dstBox.m_vMax[0] - dstBox.m_vMin[0];
            MyUnits<T> srcBoxSize = srcBox.m_vMax[0] - srcBox.m_vMin[0];
            T fAccuracy = removeUnits((dstBoxSize + srcBoxSize) / fDist);
            if (fAccuracy <= accuracyThreshold())
            {
                auto& srcNode = m_ocTree[srcNodeIndex];
                if (srcNode.m_nodeData.m_iTotalCharge != 0)
                {
                    auto& dstNode = m_ocTree[dstLeafIndex];
                    nvAssert(dstNode.getNPoints());
                    for (NvU32 uDstPoint = dstNode.getFirstPoint(); uDstPoint < dstNode.getEndPoint(); ++uDstPoint)
                    {
                        Point& dstPoint = m_points[uDstPoint];
                        rtvector<MyUnits<T>,3> vForce = coloumbLaw(dstPoint.m_vPos, dstPoint.m_iCharge, srcBoxCenter, srcNode.m_nodeData.m_iTotalCharge);
                        dstPoint.m_vForce += vForce;
                    }
                }
#if ASSERT_ONLY_CODE
                {
                    auto& dstNode = m_ocTree[dstLeafIndex];
                    m_dbgNContributions += dstNode.getNPoints() * srcNode.getNPoints();
#if RANDOM_POINT_DEBUG
                    if (m_dbgUDstPoint >= dstNode.getFirstPoint() && m_dbgUDstPoint < dstNode.getEndPoint())
                    {
                        for (NvU32 uSrcPoint = srcNode.getFirstPoint(); uSrcPoint < srcNode.getEndPoint(); ++uSrcPoint)
                        {
                            nvAssert(m_dbgSrcAccounted[uSrcPoint] == 0);
                            m_dbgSrcAccounted[uSrcPoint] = 1;
                        }
                    }
#endif
                }
#endif
                return true;
            }
        }
        auto& srcNode = m_ocTree[srcNodeIndex];
        if (!srcNode.isLeaf())
            return false;
        // if srcNode is leaf, then go over all its points and compute point-to-point interactions. as optimization, we can apply those interaction
        // in symmetric way - applying both force and counter-force at the same time. to avoid double-counting interactions, we only do that when:
        if (dstLeafIndex <= srcNodeIndex)
        {
            auto& dstNode = m_ocTree[dstLeafIndex];
            nvAssert(dstNode.getNPoints());
            for (NvU32 uSrcPoint = srcNode.getFirstPoint(); uSrcPoint < srcNode.getEndPoint(); ++uSrcPoint)
            {
                auto& srcPoint = m_points[uSrcPoint];
                if (srcPoint.m_iCharge == 0)
                    continue;
                for (NvU32 uDstPoint = dstNode.getFirstPoint(); uDstPoint < dstNode.getEndPoint(); ++uDstPoint)
                {
                    if (dstLeafIndex == srcNodeIndex && uDstPoint >= uSrcPoint) // avoid double-counting
                        continue;
                    auto& dstPoint = m_points[uDstPoint];
                    if (dstPoint.m_iCharge != 0)
                    {
                        rtvector<MyUnits<T>,3> vForce = coloumbLaw(dstPoint.m_vPos, dstPoint.m_iCharge, srcPoint.m_vPos, srcPoint.m_iCharge);
                        dstPoint.m_vForce += vForce;
                        srcPoint.m_vForce -= vForce;
#if ASSERT_ONLY_CODE
                        m_dbgNContributions += 2;
#if RANDOM_POINT_DEBUG
                        if (m_dbgUDstPoint == uDstPoint || m_dbgUDstPoint == uSrcPoint)
                        {
                            NvU32 uTmp = m_dbgUDstPoint ^ uDstPoint ^ uSrcPoint;
                            nvAssert(m_dbgSrcAccounted[uTmp] == 0);
                            m_dbgSrcAccounted[uTmp] = 1;
                        }
#endif
#endif
                    }
                }
            }
        }
        return true;
    }

private:
    void initForcesRecursive(NvU32 uNode)
    {
        auto& node = m_ocTree[uNode];
        node.m_nodeData.m_iTotalCharge = MyUnits<int>(0);
        if (node.isLeaf())
        {
            for (NvU32 uPoint = node.getFirstPoint(); uPoint < node.getEndPoint(); ++uPoint)
            {
                auto& point = m_points[uPoint];
                point.m_vForce.set(MyUnits<T>(0));
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
    BBox3<MyUnits<T>> m_bbox;
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
                NvU32 childIndex = stack.pop();
                nvAssert(childIndex == uChild);
                nvAssert(stack.getCurDepth() == dbgDepth);
                nvAssert(stack.getBox(dbgDepth) == dbgBox);
            }
            nvAssert(dbgNPoints1 == dbgNPoints2);
        }
    }
    MyUnits<T> m_fRadius = MyUnits<T>::microMeter() * 100; // it's a circle for simplicity
    std::vector<Point> m_points;
    std::vector<OcTreeNode<Neuron>> m_ocTree;
    RNGUniform m_rng;
#if ASSERT_ONLY_CODE
    NvU64 m_dbgNContributions;
#if RANDOM_POINT_DEBUG
    NvU32 m_dbgUDstPoint;
    std::vector<NvU32> m_dbgSrcAccounted;
#endif
#endif
};