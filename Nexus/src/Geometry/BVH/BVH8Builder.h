#pragma once
#include "BVH8.h"

#include <iostream>

#include "BVH.h"

class BVH8Builder
{
public:
	BVH8Builder(BVH* bvh);

	enum struct Decision
	{
		UNDEFINED = -1,
		LEAF,
		INTERNAL,
		DISTRIBUTE
	};

	struct NodeEval
	{
		// SAH cost of node n at index i
		float cost;

		// Decision made for the node
		Decision decision = Decision::UNDEFINED;

		// Left and right count if decision is DISTRIBUTE
		int leftCount, rightCount;
	};

	std::shared_ptr<BVH8> Build();
	int ComputeNodeTriCount(int nodeIdx);
	float ComputeNodeCost(uint32_t nodeIdx, int i);
	void Init();

	// Returns the indices of the node's children
	void GetChildrenIndices(uint32_t nodeIdxBvh2, uint32_t *indices, int i, int& indicesCount);
	void CollapseNode(uint32_t nodeIdxBvh2, uint32_t nodeIdxBvh8, int triBaseIdx);

private:

	// Cleaf(n)
	inline float CLeaf(const BVHNode& node, int triCount);

	// Cinternal(n)
	float CInternal(const BVHNode& node, int& leftCount, int& rightCount);

	// Cdistribute(n, j)
	float CDistribute(const BVHNode& node, int j, int& leftCount, int& rightCount);

private:
	BVH* bvh2 = nullptr;
	std::shared_ptr<BVH8> bvh8 = nullptr;

	// Optimal SAH cost C(n, i) with decisions
	std::vector<std::vector<NodeEval>> evals;

	// Number of triangles in the subtree of the node i
	std::vector<int> triCount;

	// Number of nodes already in the BVH
	uint32_t usedNodes;

	// Current base triangle index
	//uint32_t triBaseIdx = 0;
};