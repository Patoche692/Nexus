#pragma once
#include <vector>
#include "Utils/Utils.h"
#include "Geometry/AABB.h"
#include "Geometry/Ray.h"
#include "Geometry/Triangle.h"

#define BINS 8

struct BVHNode
{
	// Bounds
	float3 aabbMin, aabbMax;

	// Either the index of the left child node, or the first index in the triangles list
	union {
		uint32_t leftNode;
		uint32_t firstTriIdx;
	};

	// Number of triangles in the node (0 if not leaf)
	uint32_t triCount;

	inline __host__ __device__ bool IsLeaf() { return triCount > 0; }
	inline __host__ __device__ float Cost()
	{
		float3 diag = aabbMax - aabbMin;
		return (diag.x * diag.y + diag.y * diag.z + diag.x * diag.z) * triCount;
	}
};

class BVH
{
public:
	BVH() = default;
	BVH(std::vector<Triangle>& triangles);
	~BVH();

	void Build();

private:
	void Subdivide(uint32_t nodeIdx);
	void UpdateNodeBounds(uint32_t nodeIdx);
	float FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos);

public:
	Triangle* triangles = nullptr;
	uint32_t* triangleIdx = nullptr;
	uint32_t nodesUsed, triCount;

	BVHNode* nodes = nullptr;

	// Ray intersection (executed on the GPU)
	inline __host__ __device__ void Intersect(Ray& ray, uint32_t instanceIdx)
	{
		BVHNode* node = &nodes[0], * stack[32];
		uint32_t stackPtr = 0;

		while (1)
		{
			if (node->IsLeaf())
			{
				for (uint32_t i = 0; i < node->triCount; i++)
					triangles[triangleIdx[node->leftNode + i]].Hit(ray, instanceIdx, triangleIdx[node->leftNode + i]);

				if (stackPtr == 0)
					break;
				else
					node = stack[--stackPtr];
				continue;
			}

			BVHNode* child1 = &nodes[node->leftNode];
			BVHNode* child2 = &nodes[node->leftNode + 1];
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


