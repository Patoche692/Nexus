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
		float cost;
		Decision decision = Decision::UNDEFINED;
	};

	BVH8* Build();
	void Init();
	int ComputeTriCount(int nodeIdx);
	float ComputeNodeCost(uint32_t nodeIdx, int i);

private:

	// Cleaf(n)
	inline float CLeaf(const BVHNode& node, int triCount);

	// Cinternal(n)
	float CInternal(const BVHNode& node);

	// Cdistribute(n, j)
	inline float CDistribute(const BVHNode& node, int j);

private:
	BVH* bvh2 = nullptr;
	BVH8* bvh8 = nullptr;

	// Optimal SAH cost C(n, i) with decisions
	std::vector<std::vector<NodeEval>> evals;

	// Number of triangles in the subtree of the node i
	std::vector<int> triCount;
};