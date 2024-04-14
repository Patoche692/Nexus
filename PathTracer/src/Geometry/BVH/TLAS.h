#pragma once

#include "Utils/cuda_math.h"
#include "BVHInstance.h"
#include "Utils/Utils.h"

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
	TLAS() {};
	TLAS(BVHInstance* bvhList, int N);
	void Build();

private:
	int FindBestMatch(int N, int A);

public:

	TLASNode* nodes;
	BVHInstance* blas;
	uint32_t nodesUsed, blasCount;
	uint32_t* nodesIdx;

	inline __host__ __device__ void Intersect(Ray& ray)
	{
		ray.invDirection = 1 / ray.direction;

		TLASNode* node = &nodes[0], * stack[32];
		uint32_t stackPtr;

		while (1)
		{
			if (node->leftRight == 0)
			{
				blas[node->BLAS].Intersect(ray, node->BLAS);

				if (stackPtr == 0)
					break;
				else
					node = stack[--stackPtr];
				continue;
			}
			TLASNode* child1 = &nodes[node->leftRight & 0xffff];
			TLASNode* child2 = &nodes[node->leftRight >> 16];
			float dist1 = AABB::intersectionAABB(ray, child1->aabbMin, child1->aabbMax);
			float dist2 = AABB::intersectionAABB(ray, child2->aabbMin, child2->aabbMax);

			if (dist1 > dist2)
			{
				Utils::Swap(dist1, dist2);
				Utils::Swap(child1, child2);
			}
			if (dist1 == 1e30f)
			{
				if (stackPtr == 0)
					break;
				else
					node = stack[--stackPtr];
			}
			else
			{
				node = child1;
				if (dist2 != 1e30f)
					stack[stackPtr++] = child2;
			}
		}
	}
};
