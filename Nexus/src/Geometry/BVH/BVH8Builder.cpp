#include "BVH8Builder.h"

BVH8Builder::BVH8Builder(BVH *bvh) : bvh2(bvh)
{
    evals = std::vector<std::vector<NodeEval>>(bvh2->nodesUsed, std::vector<NodeEval>(7));
    triCount = std::vector<int>(bvh2->nodesUsed);
}

BVH8* BVH8Builder::Build()
{
    return nullptr;
}

float BVH8Builder::CLeaf(const BVHNode& node, int triCount)
{
    if (triCount > P_MAX)
        return 1.0e30f;

    AABB nodeAABB(node.aabbMin, node.aabbMax);
    return nodeAABB.Area() * triCount * C_PRIM;
}

float BVH8Builder::CDistribute(const BVHNode& node, int j)
{
    float cDistribute = 1.0e30f;
	for (int k = 0; k < j - 1; k++)
	{
        const float cLeft = ComputeNodeCost(node.leftNode, k);
        const float cRight = ComputeNodeCost(node.leftNode + 1, j - 1 - k);

        cDistribute = fmin(cDistribute, cLeft + cRight);
	}
    return cDistribute;
}

float BVH8Builder::CInternal(const BVHNode& node)
{
    AABB nodeAABB(node.aabbMin, node.aabbMax);
    return CDistribute(node, 7) + nodeAABB.Area() * C_NODE;
}

float BVH8Builder::ComputeNodeCost(uint32_t nodeIdx, int i)
{
    if (evals[nodeIdx][i].decision != Decision::UNDEFINED)
        return evals[nodeIdx][i].cost;

    const BVHNode& node = bvh2->nodes[nodeIdx];

    if (node.IsLeaf())
    {
        //triCount[nodeIdx] = node.triCount;

        // TODO: can be optimized by setting all costs for i in (0, 6) to cLeaf
        evals[nodeIdx][i].decision = Decision::LEAF;
        evals[nodeIdx][i].cost = CLeaf(node, node.triCount);

        return evals[nodeIdx][i].cost;
    }

	//triCount[nodeIdx] = triCount[node.leftNode] + triCount[node.leftNode + 1];

    // i = 1 in paper
    if (i == 0)
    {
        const float cLeaf = CLeaf(node, triCount[nodeIdx]);
        const float cInternal = CInternal(node);

        if (cLeaf < cInternal)
        {
            evals[nodeIdx][i].decision = Decision::LEAF;
            evals[nodeIdx][i].cost = cLeaf;
        }
        else
        {
            evals[nodeIdx][i].decision = Decision::INTERNAL;
            evals[nodeIdx][i].cost = cInternal;
        }
        return evals[nodeIdx][i].cost;
    }

    // i in (2, 7) in paper
    const float cDistribute = CDistribute(node, i);
    const float cFewerRoots = ComputeNodeCost(nodeIdx, i - 1);

    if (cDistribute < cFewerRoots)
    {
        evals[nodeIdx][i].decision = Decision::DISTRIBUTE;
        evals[nodeIdx][i].cost = cDistribute;
    }
    else
    {
        evals[nodeIdx][i].decision = evals[nodeIdx][i - 1].decision;
        evals[nodeIdx][i].cost = cFewerRoots;
    }

    return evals[nodeIdx][i].cost;
}

void BVH8Builder::Init()
{
    ComputeTriCount(0);
    float rootCost = ComputeNodeCost(0, 0);
	std::cout << rootCost << std::endl;
}

int BVH8Builder::ComputeTriCount(int nodeIdx)
{
    BVHNode& node = bvh2->nodes[nodeIdx];

    if (node.IsLeaf())
        triCount[nodeIdx] = node.triCount;
    else
		triCount[nodeIdx] = ComputeTriCount(node.leftNode) + ComputeTriCount(node.leftNode + 1);

    return triCount[nodeIdx];
}
