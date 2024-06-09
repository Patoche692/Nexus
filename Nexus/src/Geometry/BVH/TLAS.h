#pragma once

#include <thrust/device_vector.h>
#include "Utils/cuda_math.h"
#include "BVHInstance.h"
#include "Utils/Utils.h"
#include "Cuda/BVH/TLAS.cuh"

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
	TLAS(const std::vector<BVHInstance>& bvhList);
	void Build();

	void UpdateDeviceData();
	D_TLAS ToDevice();

private:
	int FindBestMatch(int N, int A);

public:

	std::vector<TLASNode> nodes;
	std::vector<BVHInstance> blas;
	std::vector<uint32_t> instancesIdx;

	// Device members
	thrust::device_vector<D_TLASNode> deviceNodes;
	thrust::device_vector<D_BVHInstance> deviceBlas;
};
