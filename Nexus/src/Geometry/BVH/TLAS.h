#pragma once

#include "Memory/Device/DeviceVector.h"
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
	TLAS(const std::vector<BVHInstance>& bvhList, const std::vector<BVH8>& bvhs);
	void Build();

	void UpdateDeviceData();
	static D_TLAS ToDevice(const TLAS& tlas);

private:
	int FindBestMatch(int N, int A);

private:

	std::vector<TLASNode> m_Nodes;
	std::vector<BVHInstance> m_Blas;
	std::vector<uint32_t> m_InstancesIdx;
	std::vector<BVH8> m_Bvhs;

	// Device members
	DeviceVector<TLASNode, D_TLASNode> m_DeviceNodes;
	DeviceVector<BVHInstance, D_BVHInstance> m_DeviceBlas;
	DeviceVector<BVH8, D_BVH8> m_DeviceBvhs;
};
