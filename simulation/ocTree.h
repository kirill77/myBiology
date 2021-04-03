#pragma once

#include "mybasics.h"

template <class Access> // implements all access to outside world required by this class
struct OcTreeNode
{
    typedef OcTreeNode<Access> NodeType;

    OcTreeNode() : m_uFirstChild(~0U) { }
    ~OcTreeNode() { }

    bool isLeaf() const { return m_uEndPoint != ~0U; }

    void initLeaf(NvU32 firstPoint, NvU32 endPoint)
    {
        m_uFirstPoint = firstPoint;
        m_uEndPoint = endPoint;
    }

    bool split(const float3 &vCenter, PointArray &points, NodeArray &nodes)
    {
        if (!m_isLeaf || m_nPoints == 0) return false;
       
        NvU32 fc = m_nodes.size();
        m_nodes.resize(fc + 8);

        nodes[fc + 0].initLeaf(looseSort(points, m_uFirstPoint, m_uEndPoint, vCenter, 2), m_uEndPoint); // split in z
        nodes[fc + 4].initLeaf(fc[0].m_uEndPoint, m_uEndPoint); // just initialize

        nodes[fc + 2].m_uFirstPoint = looseSort(points, m_uFirstPoint, nodes[fc + 4].m_uFirstPoint, vCenter, 1); // split in y
        nodes[fc + 6].m_uFirstPoint = looseSort(nodes[fc + 4].m_uFirstPoint, m_uEndPoint, vCenter, 1); // split in y

        nodes[fc + 1].initLeaf(looseSort(points, m_uFirstPoint, nodes[fc + 2].m_uFirstPoint, vCenter, 0); // split in x
        nodes[fc + 3].initLeaf(looseSort(points, notes[fc + 2].m_uFirstNode, nodes[fc + 4].m_uFirstPoint, vCenter, 0); // split in x
        nodes[fc + 5].initLeaf(looseSort(points, nodes[fc + 4], nodes[fc + 6].m_uFirstPoint, vCenter, 0); // split in x
        nodes[fc + 7].initLeaf(looseSort(points, notes[fc + 6].m_uFirstNode, m_uEndPoint, vCenter, 0); // split in x

        m_uEndPoint = ~0U;

        return true;
    }

    enum ACCURACY_DECISION { ACCURACY_TOO_CLOSE, ACCURACY_MATCH, ACCURACY_TOO_FAR };

    struct BoxIterator
    {
        BoxIterator(NvU32 uRootNode, const BBox3f& rootBox);
        BoxIterator(const BoxIterator& other);
        void descend(NodeArray& nodeArray, NvU32 childIndex);
        NvU32 ascend(); // returns index of the child where we've been
        NvU32 getCurNode() const { return nodesStack[m_curDepth]; }
        NvU32 getCurDepth() const { return m_curDepth; }
        const BBox3f& getCurBox() const { return m_curBox; }
        ACCURACY_DECISION computeAccuracyDecision(const Iterator& other, NvU32 nAccuracyDist);
    private:
        NvU32 m_uBox[3]; // starts with 0,0,0, *= 2 when we descend, += 1 when we shift
        float3 m_boundsStack[32];
        BBox3f m_curBox;
        NvU32 m_nodesStack[32];
        NvU32 m_curDepth;
    };

    static double beforeComputePointForces(NodeType& root, NodeArray& nodes, PointArray& points)
    {
        root.m_nodeData.setForce(float3(0));
        double fCharge = 0;
        if (root.isLeaf())
        {

        }
    }
    // assuming each point has scalar charge and there is a force acting between each pair of charges, compute cumulative force acting on each point.
    // accuracy is understood like this: if two octree boxes are of the same size and distance between them is >= nAccuracyDist, we compute box<->box
    // force instead of point<->point force
    static void computePointForces(NvU32 rootIndex, Access &access)
    {
        bool bIt1NextSucceeded = true;
        for (BoxIterator it1(rootIndex, access); bIt1NextSucceeded; bIt1NextSucceeded = it1.next())
        {
            if (it1.isEmptyNode())
                continue;
            NodeType& curNode = it1.getCurNode();
            if (curNode.isLeaf())
            {
                access.addPoint2PointContributions(curNode.m_uFirstPoint, curNode.m_uEndPoint);
            }
            BoxIterator it2(it1);
            if (!it2.nextNoDescendent())
                continue;
            bool bIt2NextSuceeded = true;
            for ( ; bIt2NextSuceeded; )
            {
                if (it2.isEmptyNode())
                {
                    bIt2NextSuceeded = it2.next();
                    continue;
                }
                if (it2.isBox2BoxAccuracyMatch(it1))
                {
                    access.addNode2NodeContributions(it1.getNodeIndex(), it1.getCurBox(), it2.getNodeIndex(), it2.getCurBox());
                    bIt2NextSuceeded = it2.nextNoDescendent();
                    continue;
                }
                if (it2.isLeaf())
                {
                    const NodeType& node1 = it1.getCurNode();
                    const NodeType& node2 = it2.getCurNode();
                    for (NvU32 uPoint2 = node2.m_firstPoint; uPoint2 < node2.m_endPoint; ++uPoint2)
                    {
                        if (it1.isBox2PointAccuracyMatch(uPoint2))
                        {
                            access.addNode2PointContribution(it1.getNodeIndex(), it1.getCurBox(), uPoint2);
                        }
                        else
                        {
                            for (NvU32 uPoint1 = node1.m_firstPoint; uPoint1 < node1.m_uEndPoint; ++uPoint1)
                            {
                                access.addPoint2PointContribution(uPoint1, uPoint2);
                            }
                        }
                    }
                    bIt2NextSuceeded = it2.next();
                    continue;
                }
                bIt2NextSuceeded = it2.next();
            }
    }

    NodeData m_nodeData;

private:
    // returns index of first point for which points[u][uDim] >= vCenter[uDim]
    static NvU32 looseSort(PointArray& points, NvU32 uBegin, NvU32 uEnd, const float3& vCenter, NvU32 uDim)
    {
        for (float fSplit = vCenter[uDim]; ; ++uBegin)
        {
            nvAssert(uBegin <= uEnd);
            if (uBegin == uEnd)
                return uEnd;
            if (points[uBegin][uDim] < fSplit)
                continue;
            // search for element with which we can swap
            for (--uEnd; ; --uEnd)
            {
                nvAssert(uBegin <= uEnd);
                if (uBegin == uEnd)
                    return uEnd;
                if (points[uEnd][uDim] < fSplit)
                    break;
            }
            points.swap(uBegin, uEnd);
        }
    }
    union
    {
        NvU32 m_uFirstChild; // index into NodeAllocator
        NvU32 m_uFirstPoint; // index into PointAllocator
    };
    NvU32 m_uEndPoint = ~0U; // inner node would have this set to ~0U

    // implementation of BoxIterator methods
    BoxIterator::BoxIterator(NvU32 uRootNode, const BBox3f& rootBox)
    {
        uBox[0] = uBox[1] = uBox[2] = uDepth = 0;
        nodesStack[0] = uRootNode;
        curBox = rootBox;
    }
    BoxIterator::BoxIterator(const BoxIterator& other)
    {
        uBox[0] = other.uBox[0];
        uBox[1] = other.uBox[1];
        uBox[2] = other.uBox[2];
        for (NvU32 u = 0; u < other.uDepth; ++u)
        {
            boundsStack[u] = other.boundsStack[u];
            nodesStack[u] = other.nodesStack[u];
        }
        curBox = other.getCurBox;
        uDepth = other.uDepth;
        nodesStack[uDepth] = other.nodesStack[uDepth];
    }
    void BoxIterator::descend(NodeArray& nodeArray, NvU32 childIndex)
    {
        nvAssert(childIndex < 8 && uDepth < 31);
        for (NvU32 uDim = 0; uDim < 3; ++uDim)
        {
            NvU32 dimBit = ((childIndex >> uDim) & 1);
            uBox[uDim] = (uBox[uDim] << 1) | dimBit;
            boundsStack[uDepth] = curBox[dimBit ^ 1][uDim];
            curBox[dimBit ^ 1][uDim] = (curBox[0][uDim] + curBox[1][uDim]) / 2;
        }
        nodesStack[uDepth + 1] = nodeArray[nodesStack[uDepth]].getChildren();
        ++uDepth;
    }
    NvU32 BoxIterator::ascend()
    {
        --uDepth;
        NvU32 dimBit = uBox[0] & 1;
        curBox[dimBit ^ 1][0] = boundsStack[uDepth][0];
        NvU32 uChild = dimBit;
        for (NvU32 uDim = 1; uDim < 3; ++uDim)
        {
            NvU32 dimBit = uBox[uDim] & 1;
            curBox[dimBit ^ 1][uDim] = boundsStack[uDepth][uDim];
            uChild = (uChild << 1) | dimBit;
        }
        return uChild;
    }
    ACCURACY_DECISION BoxIterator::computeAccuracyDecision(const Iterator& other, NvU32 nAccuracyDist)
    {
        auto uDist = c1.uBox[0] >= c2.uBox[0] ? c1.uBox[0] - c2.uBox[0] : c2.uBox[0] - c1.uBox[0]);
        for (NvU32 uDim = 1; uDim < 3; ++uDim)
        {
            uDist = std::max(uDist, c1.uBox[uDim] >= c2.uBox[uDim] ? c1.uBox[uDim] - c2.uBox[uDim] : c2.uBox[uDim] - c1.uBox[uDim]);
        }
        if (uDist < nAccuracyDecision)
            return ACCURACY_TOO_CLOSE;
        // the goal here is to avoid counting same volume twice. ACCURACY_TOO_FAR means that this volume must have been counted on the coarser level
        return uDist / 2 >= nAccuracyDist ? ACCURACY_TOO_FAR : ACCURACY_MATCH;
    }
};
