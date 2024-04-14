#pragma once

#include "Utils/cuda_math.h"
#include "BVHInstance.h"

struct TLASNode
{
	float3 aabbMin;
	float3 aabbMax;
	uint32_t leftRight;
	uint32_t BLAS;
	bool IsLeaf() { return leftRight == 0; }
};

class TLAS
{
public:
	TLAS() = default;
	TLAS(BVHInstance* bvhList, int N);
	void Build();
	void Intersect(Ray& ray);
private:
	int FindBestMatch(int* list, int N, int A);

	TLASNode* nodes = nullptr;
	BVHInstance* blas = nullptr;
	uint32_t nodesUsed, blasCount;
};
