#pragma once
#include "BVH8.h"

#include <iostream>

#include "BVH.h"

#define N_Q 8	// Number of bits used to store the childs' AABB coordinates

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

	BVH8* Build();
	int ComputeNodeTriCount(int nodeIdx);
	float ComputeNodeCost(uint32_t nodeIdx, int i);
	void Init();
	void CollapseNode(uint32_t nodeIdxBvh2, int i, uint32_t nodeIdxBvh8);

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
	uint32_t usedNodes = 1;

	// Current base triangle index
	uint32_t triBaseIdx = 0;
};