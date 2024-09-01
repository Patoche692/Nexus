#pragma once

#include "Device/DeviceVector.h"
#include "Utils/cuda_math.h"
#include "BVHInstance.h"
#include "Utils/Utils.h"
#include "Cuda/BVH/TLAS.cuh"

struct TLASNode
{
	float3 aabbMin;
	float3 aabbMax;
	uint32_t left;
	uint32_t right;
	union
	{
		// If leaf: index of BLAS, if internal node: number of BLAS
		uint32_t blasCount;
		uint32_t blasIdx;
	};
	inline bool IsLeaf() const { return left == 0; }
};

struct TLAS
{
	TLAS() = default;
	TLAS(const std::vector<BVHInstance>& instancesList, const std::vector<BVH8>& bvhList);
	void Build();

	void UpdateDeviceData();
	void SetBVHInstances(const std::vector<BVHInstance>& instances) { bvhInstances = instances; }

	static D_TLAS ToDevice(const TLAS& tlas);

	std::vector<BVHInstance>& GetInstances() { return bvhInstances; }

	int FindBestMatch(int N, int A);

	std::vector<TLASNode> nodes;
	std::vector<BVHInstance> bvhInstances;
	std::vector<uint32_t> instancesIdx;
	std::vector<BVH8> bvhs;

	// Device members
	DeviceVector<TLASNode, D_TLASNode> deviceNodes;
	DeviceVector<BVHInstance, D_BVHInstance> deviceBlas;
	DeviceVector<BVH8, D_BVH8> deviceBvhs;
};
