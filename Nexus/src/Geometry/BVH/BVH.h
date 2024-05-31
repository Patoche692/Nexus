#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include <cudart_platform.h>
#include "Utils/Utils.h"
#include "Geometry/AABB.h"
#include "Geometry/Ray.h"
#include "Geometry/Triangle.h"

// Standard SAH-Based BVH with binned building adapted from Jacco Bikker's guides
// See https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
// The main difference is that the leaves contain only one primitive to allow
// for collapsing the nodes to construct more advanced BVHs (e.g. compressed wide BVHs)

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

	inline __host__ __device__ bool IsLeaf() const { return triCount > 0; }
	inline __host__ __device__ float Cost() const
	{
		float3 diag = aabbMax - aabbMin;
		return (diag.x * diag.y + diag.y * diag.z + diag.x * diag.z) * triCount;
	}
};

typedef struct Bin 
{
	AABB bounds;
	int triCount = 0;
} Bin;

class BVH
{
public:
	BVH() = default;
	BVH(std::vector<Triangle>& triangles);
	~BVH();

	void Build();

private:
	void SplitNodeInHalf(BVHNode& node);
	void Subdivide(uint32_t nodeIdx);
	void UpdateNodeBounds(uint32_t nodeIdx);
	float FindBestSplitPlane(const BVHNode& node, int& axis, double& splitPos);

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
				//int a = __uint_as_float(stackPtr);
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


