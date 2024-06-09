#include "BVH8.h"
#include <numeric>

BVH8::BVH8(const std::vector<Triangle>& tri)
{
	triangles = tri;
	deviceTriangles = tri;
	triangleIdx = std::vector<uint32_t>(triangles.size());
}

void BVH8::Init()
{
	// Fill the indices with integers starting from 0
	std::iota(triangleIdx.begin(), triangleIdx.end(), 0);
}

void BVH8::UpdateDeviceData()
{
	deviceNodes = nodes;
	deviceTriangleIdx = triangleIdx;
}

D_BVH8 BVH8::ToDevice()
{
	D_BVH8 deviceBvh;
	deviceBvh.triangles = thrust::raw_pointer_cast(deviceTriangles.data());
	deviceBvh.triangleIdx = thrust::raw_pointer_cast(deviceTriangleIdx.data());
	deviceBvh.nodes = thrust::raw_pointer_cast(deviceNodes.data());
	deviceBvh.nodesUsed = nodes.size();
	deviceBvh.triCount = triangles.size();
	return deviceBvh;
}

