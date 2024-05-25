#include "BVH8Builder.h"

BVH8Builder::BVH8Builder(BVH *bvh) : bvh2(bvh), bvh8(std::make_shared<BVH8>(bvh))
{
    evals = std::vector<std::vector<NodeEval>>(bvh2->nodesUsed, std::vector<NodeEval>(7));
    triCount = std::vector<int>(bvh2->nodesUsed);
}

std::shared_ptr<BVH8> BVH8Builder::Build()
{
    usedNodes = 1;
    CollapseNode(0, 0, 0);
    std::cout << "Used nodes: " << usedNodes << std::endl;
    return bvh8;
}

float BVH8Builder::CLeaf(const BVHNode& node, int triCount)
{
    if (triCount > P_MAX)
		return 1.0e30f;

    AABB nodeAABB(node.aabbMin, node.aabbMax);
    return nodeAABB.Area() * triCount * C_PRIM;
}

float BVH8Builder::CDistribute(const BVHNode& node, int j, int& leftCount, int& rightCount)
{
    float cDistribute = 1.0e30f;

    // k in (1 .. j) in the paper
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

    // i = 1 in the paper
    if (i == 0)
    {
        int leftCount, rightCount;
        const float cLeaf = CLeaf(node, triCount[nodeIdx]);
        const float cInternal = CInternal(node, leftCount, rightCount);

        if (cLeaf < cInternal)
        {
            evals[nodeIdx][i].decision = Decision::LEAF;
            evals[nodeIdx][i].cost = cLeaf;
        }
        else
        {
            evals[nodeIdx][i].decision = Decision::INTERNAL;
            evals[nodeIdx][i].cost = cInternal;
			evals[nodeIdx][i].leftCount = leftCount;
			evals[nodeIdx][i].rightCount = rightCount;
        }
        return evals[nodeIdx][i].cost;
    }

    // i in (2 .. 7) in the paper
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
        evals[nodeIdx][i] = evals[nodeIdx][i - 1];

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

void BVH8Builder::GetChildrenIndices(uint32_t nodeIdxBvh2, int* indices, int i, int& indicesCount)
{
	const NodeEval& eval = evals[nodeIdxBvh2][i];

    // If in the first call the node is a leaf, return
	if (eval.decision == Decision::LEAF)
	{
		indices[indicesCount++] = nodeIdxBvh2;
		return;
	}

	// Decision is either INTERNAL or DISTRIBUTE
	const BVHNode& node = bvh2->nodes[nodeIdxBvh2];

	const int leftCount = eval.leftCount;
	const int rightCount = eval.rightCount;

	// Retreive the decision for the left and right childs
	const NodeEval& leftEval = evals[node.leftNode][leftCount];
	const NodeEval& rightEval = evals[node.leftNode + 1][rightCount];

	// Recurse in child nodes if we need to distribute
	if (leftEval.decision == Decision::DISTRIBUTE)
		GetChildrenIndices(node.leftNode, indices, leftCount, indicesCount);
	else
		indices[indicesCount++] = node.leftNode;   // We reached a BVH8 internal node or leaf => stop recursion

	if (rightEval.decision == Decision::DISTRIBUTE)
		GetChildrenIndices(node.leftNode + 1, indices, rightCount, indicesCount);
	else
		indices[indicesCount++] = node.leftNode + 1;   // We reached a BVH8 internal node or leaf => stop recursion
}

void BVH8Builder::OrderChildren(uint32_t nodeIdxBvh2, int* childrenIndices)
{
    const BVHNode& parentNode = bvh2->nodes[nodeIdxBvh2];
    const float3 parentCentroid = (parentNode.aabbMax + parentNode.aabbMin) * 0.5f;

    // Fill the table cost(c, s)
    float cost[8][8];
    int childCount = 0;

    for (int c = 0; c < 8; c++)
    {
        // If no more children, break
        if (childrenIndices[c] == -1)
            break;

        for (int s = 0; s < 8; s++)
        {
            // Ray direction
            const float dsx = (s & 0b100) ? -1.0f : 1.0f;
            const float dsy = (s & 0b010) ? -1.0f : 1.0f;
            const float dsz = (s & 0b001) ? -1.0f : 1.0f;
            const float3 ds = make_float3(dsx, dsy, dsz);

            const BVHNode& childNode = bvh2->nodes[childrenIndices[c]];
            const float3 centroid = (childNode.aabbMin + childNode.aabbMax) * 0.5f;
            cost[c][s] = dot(centroid - parentCentroid, ds);
        }
        childCount++;
    }

    // Greedy ordering

    bool slotAssigned[8] = { 0 };
    int assignment[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };

    while (true)
    {
		float minCost = std::numeric_limits<float>::max();
        int assignedNode = -1, assignedSlot = -1;

        for (int c = 0; c < childCount; c++)
        {
            // If node already assigned, skip
            if (assignment[c] != -1)
                continue;

            for (int s = 0; s < 8; s++)
            {
                // If slot already used, skip
                if (slotAssigned[s])
                    continue;

                if (cost[c][s] < minCost)
                {
                    minCost = cost[c][s];
                    assignedNode = c;
                    assignedSlot = s;
                }
            }
        }

        // If all the nodes have been assigned
        if (assignedNode == -1)
            break;

        // Assign the node to the specific position
        assignment[assignedNode] = assignedSlot;
        slotAssigned[assignedSlot] = true;
    }

    int indicesCpy[8];
    memcpy(indicesCpy, childrenIndices, 8 * sizeof(int));

    for (int i = 0; i < 8; i++)
        childrenIndices[i] = -1;

    // Reorder the nodes
    for (int i = 0; i < childCount; i++)
        childrenIndices[assignment[i]] = indicesCpy[i];

}

void BVH8Builder::CollapseNode(uint32_t nodeIdxBvh2, uint32_t nodeIdxBvh8, int triBaseIdx)
{
    const BVHNode& bvh2Node = bvh2->nodes[nodeIdxBvh2];

    BVH8Node bvh8Node;

    const float denom = 1 / (powf(2, N_Q) - 1);
    
    // e along each axis
    const float ex = ceilf(log2f((bvh2Node.aabbMax.x - bvh2Node.aabbMin.x) * denom));
    const float ey = ceilf(log2f((bvh2Node.aabbMax.y - bvh2Node.aabbMin.y) * denom));
    const float ez = ceilf(log2f((bvh2Node.aabbMax.z - bvh2Node.aabbMin.z) * denom));

    bvh8Node.e[0] = static_cast<byte>(ex);
    bvh8Node.e[1] = static_cast<byte>(ey);
    bvh8Node.e[2] = static_cast<byte>(ez);

    bvh8Node.childBaseIdx = usedNodes;
    bvh8Node.triangleBaseIdx = triBaseIdx;

    bvh8Node.p = bvh2Node.aabbMin;

    int childrenIndices[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int indicesCount = 0;

    // Fill the array of children indices
	GetChildrenIndices(nodeIdxBvh2, childrenIndices, 0, indicesCount);

    // Order the children according to the octant traversal order
    OrderChildren(nodeIdxBvh2, childrenIndices);

    // Sum of triangles number in the node
    int nTrianglesTotal = 0;

	const float scaleX = 1 / pow(2, bvh8Node.e[0]);
	const float scaleY = 1 / pow(2, bvh8Node.e[1]);
	const float scaleZ = 1 / pow(2, bvh8Node.e[2]);

    for (int i = 0; i < 8; i++)
    {
        if (childrenIndices[i] == -1)
        {
            // Empty child slot, set meta to 0
            bvh8Node.meta[i] = 0;
            continue;
        }
        else
        {
			// Since the children are either internal or leaf nodes, we take their evaluation for i = 1
			const NodeEval& eval = evals[childrenIndices[i]][0];

            // Encode the child's bounding box origin
            bvh8Node.qlox[i] = static_cast<byte>((bvh2Node.aabbMin.x - bvh8Node.p.x) / scaleX);
            bvh8Node.qloy[i] = static_cast<byte>((bvh2Node.aabbMin.y - bvh8Node.p.y) / scaleY);
            bvh8Node.qloz[i] = static_cast<byte>((bvh2Node.aabbMin.z - bvh8Node.p.z) / scaleZ);

            // Encode the child's bounding box end point
            bvh8Node.qhix[i] = static_cast<byte>((bvh2Node.aabbMax.x - bvh8Node.p.x) / scaleX);
            bvh8Node.qhiy[i] = static_cast<byte>((bvh2Node.aabbMax.y - bvh8Node.p.y) / scaleY);
            bvh8Node.qhiz[i] = static_cast<byte>((bvh2Node.aabbMax.z - bvh8Node.p.z) / scaleZ);

            if (eval.decision == Decision::INTERNAL)
            {
                // High 3 bits to 001
                bvh8Node.meta[i] = 0b00100000;
                // Low 5 bits to 24 + child index
                bvh8Node.meta[i] |= 24 + i;
                // Set the child node as an internal node in the imask field
                bvh8Node.imask |= 1 << i;
            }
            else if (eval.decision == Decision::LEAF)
            {
                const int nTriangles = triCount[childrenIndices[i]];
                assert(nTriangles <= P_MAX);

                // High 3 bits store the number of triangles in unary encoding
                for (int j = 0; j < nTriangles; j++)
                {
                    bvh8Node.meta[i] |= 1 << (j + 5);
                }
                nTrianglesTotal += nTriangles;
                assert(nTrianglesTotal <= 23);

                // Low 5 bits store the index of first triangle relative to the triangle base index
                bvh8Node.meta[i] |= nTrianglesTotal;
                // Low 5 bits to 24 + child index
                bvh8Node.meta[i] |= 24 + i;
                // Set the child node as an internal node in the imask field
                bvh8Node.imask |= 1 << i;
            }
        }
        usedNodes++;
    }
	bvh8->nodes[nodeIdxBvh8] = bvh8Node;

    nTrianglesTotal = 0;

    // Recursively collapse internal children nodes
    for (int i = 0; i < 8; i++)
    {
        if (childrenIndices[i] == -1)
            continue;

		nTrianglesTotal += triCount[childrenIndices[i]];

        NodeEval& eval = evals[childrenIndices[i]][0];

        if (eval.decision == Decision::INTERNAL)
			CollapseNode(childrenIndices[i], bvh8Node.childBaseIdx + i, triBaseIdx + nTrianglesTotal);
    }
}

