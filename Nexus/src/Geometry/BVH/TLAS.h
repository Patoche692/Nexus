#pragma once

#include "Utils/cuda_math.h"
#include "BVHInstance.h"
#include "Utils/Utils.h"

struct TLASNode
{
	float3 aabbMin;
	float3 aabbMax;
	uint32_t leftRight;
	uint32_t blasIdx;
	inline bool IsLeaf() { return leftRight == 0; }
};

class TLAS
{
public:
	TLAS() = default;
	TLAS(BVHInstance* bvhList, int N);
	void Build();
	TLAS* ToDevice();

private:
	int FindBestMatch(int N, int A);

public:

	TLASNode* nodes;
	BVHInstance* blas;
	uint32_t nodesUsed, blasCount;
	uint32_t* nodesIdx;

};
