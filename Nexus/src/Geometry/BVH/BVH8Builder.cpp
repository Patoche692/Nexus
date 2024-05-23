#include "BVH8Builder.h"

BVH8Builder::BVH8Builder(BVH *bvh) : bvh2(bvh), bvh8(std::make_shared<BVH8>(bvh))
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
    {
        if (node.IsLeaf())
            std::cout << "Grave pb: " << node.triCount << std::endl;
		return 1.0e30f;
    }

    AABB nodeAABB(node.aabbMin, node.aabbMax);
    return nodeAABB.Area() * triCount * C_PRIM;
}

float BVH8Builder::CDistribute(const BVHNode& node, int j, int& leftCount, int& rightCount)
{
    float cDistribute = 1.0e30f;

    // k in (1 .. j) in paper
	for (int k = 0; k < j - 1; k++)
	{
        const float cLeft = ComputeNodeCost(node.leftNode, k);
        const float cRight = ComputeNodeCost(node.leftNode + 1, j - 1 - k);

        if (cLeft + cRight < cDistribute)
        {
            cDistribute = cLeft + cRight;
            leftCount = k;
            rightCount = j - 1 - k;
        }
	}
    return cDistribute;
}

float BVH8Builder::CInternal(const BVHNode& node, int& leftCount, int&rightCount)
{
    AABB nodeAABB(node.aabbMin, node.aabbMax);
    return CDistribute(node, 7, leftCount, rightCount) + nodeAABB.Area() * C_NODE;
}

float BVH8Builder::ComputeNodeCost(uint32_t nodeIdx, int i)
{
    if (evals[nodeIdx][i].decision != Decision::UNDEFINED)
        return evals[nodeIdx][i].cost;

    const BVHNode& node = bvh2->nodes[nodeIdx];

    if (node.IsLeaf())
    {
        // TODO: can be optimized by setting all costs for i in (0 .. 6) to cLeaf
        evals[nodeIdx][i].decision = Decision::LEAF;
        evals[nodeIdx][i].cost = CLeaf(node, node.triCount);

        return evals[nodeIdx][i].cost;
    }

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

    // i in (2 .. 7) in paper
    int leftCount, rightCount;
    const float cDistribute = CDistribute(node, i, leftCount, rightCount);
    const float cFewerRoots = ComputeNodeCost(nodeIdx, i - 1);

    if (cDistribute < cFewerRoots)
    {
        evals[nodeIdx][i].decision = Decision::DISTRIBUTE;
        evals[nodeIdx][i].cost = cDistribute;
        evals[nodeIdx][i].leftCount = leftCount;
        evals[nodeIdx][i].rightCount = rightCount;
    }
    else
    {
        evals[nodeIdx][i].decision = evals[nodeIdx][i - 1].decision;
        evals[nodeIdx][i].cost = cFewerRoots;
    }

    return evals[nodeIdx][i].cost;
}

int BVH8Builder::ComputeNodeTriCount(int nodeIdx)
{
    BVHNode& node = bvh2->nodes[nodeIdx];

    if (node.IsLeaf())
        triCount[nodeIdx] = node.triCount;
    else
		triCount[nodeIdx] = ComputeNodeTriCount(node.leftNode) + ComputeNodeTriCount(node.leftNode + 1);

    return triCount[nodeIdx];
}

void BVH8Builder::Init()
{
    ComputeNodeTriCount(0);
    float rootCost = ComputeNodeCost(0, 0);
	std::cout << rootCost << std::endl;
}

void BVH8Builder::CollapseNode(uint32_t nodeIdxBvh2, int i, uint32_t nodeIdxBvh8)
{
    const BVHNode& bvh2Node = bvh2->nodes[nodeIdxBvh2];

    const NodeEval& eval = evals[nodeIdxBvh2][i];

    BVH8Node bvh8Node = { };

    const float denom = 1 / (powf(2, N_Q) - 1);
    
    // ei
    const float ex = ceilf(log2f((bvh2Node.aabbMax.x - bvh2Node.aabbMin.x) * denom));
    const float ey = ceilf(log2f((bvh2Node.aabbMax.y - bvh2Node.aabbMin.y) * denom));
    const float ez = ceilf(log2f((bvh2Node.aabbMax.z - bvh2Node.aabbMin.z) * denom));

    bvh8Node.e[0] = static_cast<byte>(ex);
    bvh8Node.e[1] = static_cast<byte>(ey);
    bvh8Node.e[2] = static_cast<byte>(ez);

    bvh8Node.childBaseIdx = usedNodes;
    bvh8Node.triangleBaseIdx = triBaseIdx;

    bvh8Node.p = bvh2Node.aabbMin;

    switch (eval.decision)
	{
	case Decision::LEAF:
        int cPrim = triCount[nodeIdxBvh2];
        for (int i = 0; i < 8; i++)
        {
            bvh8Node.meta[i] = ;
        }


		break;
	case Decision::INTERNAL:
        int leftCount = eval.leftCount, rightCount = eval.rightCount;



		break;
	case Decision::DISTRIBUTE:

		break;
	}

}

