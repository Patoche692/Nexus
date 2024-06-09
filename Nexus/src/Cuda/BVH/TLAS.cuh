#pragma once

#include "BVHInstance.cuh"
#include "Cuda/Geometry/Ray.cuh"

struct D_TLASNode
{
	float3 aabbMin;
	float3 aabbMax;
	uint32_t leftRight;
	uint32_t blasIdx;
	inline __device__ bool IsLeaf() { return leftRight == 0; }
};

struct D_TLAS
{
	D_TLASNode* nodes;
	D_BVHInstance* blas;
};

