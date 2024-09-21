#pragma once

#include "Device/DeviceVector.h"
#include "Utils/cuda_math.h"
#include "BVHInstance.h"
#include "Utils/Utils.h"

struct TLASNode
{
	float3 aabbMin;
	float3 aabbMax;
	uint32_t left;
	uint32_t right;
	uint32_t blasCount;
	uint32_t blasIdx;
	inline bool IsLeaf() const { return left == 0; }
};

struct TLAS
{
	TLAS() = default;
	TLAS(const std::vector<BVHInstance>& instancesList, const std::vector<BVH8>& bvhList);
	void Build();
	void Convert();

	void UpdateDeviceData();
	void SetBVHInstances(const std::vector<BVHInstance>& instances) { bvhInstances = instances; }

	std::vector<BVHInstance>& GetInstances() { return bvhInstances; }

	int FindBestMatch(int N, int A);

	std::vector<TLASNode> nodes;
	std::vector<BVHInstance> bvhInstances;
	std::vector<uint32_t> instancesIdx;
	BVH8 bvh8;

	// Device members
	DeviceVector<BVHInstance, D_BVHInstance> deviceBlas;

	DeviceInstance<D_BVHInstance*> deviceBlasAddress;
	DeviceInstance<BVH8, D_BVH8> deviceBvh8;
};
