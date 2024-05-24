#pragma once

#include "Geometry/BVH/BVH.h"
#include "Geometry/Ray.h"

inline __device__ void IntersectBVH2(const BVH& bvh, Ray& ray, const uint32_t instanceIdx)
{
	BVHNode* node = &bvh.nodes[0], * stack[32];
	uint32_t stackPtr = 0;

	while (1)
	{
		if (node->IsLeaf())
		{
			for (uint32_t i = 0; i < node->triCount; i++)
				bvh.triangles[bvh.triangleIdx[node->leftNode + i]].Hit(ray, instanceIdx, bvh.triangleIdx[node->leftNode + i]);

			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
			continue;
		}

		BVHNode* child1 = &bvh.nodes[node->leftNode];
		BVHNode* child2 = &bvh.nodes[node->leftNode + 1];
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
