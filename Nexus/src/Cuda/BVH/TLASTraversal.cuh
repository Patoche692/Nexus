#pragma once

#include "TLAS.cuh"
#include "BVHInstanceTraversal.cuh"
#include "Cuda/Geometry/AABB.cuh"

inline __device__ void IntersectTLAS(const D_TLAS& tlas, D_Ray& ray)
{
	D_TLASNode* node = &tlas.nodes[0], * stack[16];
	uint32_t stackPtr = 0;

	while (1)
	{
		if (node->IsLeaf())
		{
			IntersectBVHInstance(tlas.blas[node->blasIdx], ray, node->blasIdx);

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
			continue;
		}
		D_TLASNode* child1 = &tlas.nodes[node->leftRight & 0xffff];
		D_TLASNode* child2 = &tlas.nodes[node->leftRight >> 16];
		float dist1 = D_AABB::IntersectionAABB(ray, child1->aabbMin, child1->aabbMax);
		float dist2 = D_AABB::IntersectionAABB(ray, child2->aabbMin, child2->aabbMax);

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