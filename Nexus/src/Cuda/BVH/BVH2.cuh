#pragma once

#include <cuda_runtime_api.h>
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Geometry/Triangle.cuh"
#include "Cuda/Geometry/AABB.cuh"

struct D_BVH2Node
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

	inline __device__ bool IsLeaf() const { return triCount > 0; }
	inline __device__ float Cost() const
	{
		float3 diag = aabbMax - aabbMin;
		return (diag.x * diag.y + diag.y * diag.z + diag.x * diag.z) * triCount;
	}
};

struct D_BVH2
{
	D_Triangle* triangles = nullptr;
	uint32_t* triangleIdx = nullptr;
	uint32_t nodesUsed, triCount;
	D_BVH2Node* nodes = nullptr;
};

