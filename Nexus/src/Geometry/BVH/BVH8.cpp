#include "BVH8.h"
#include <numeric>

BVH8::BVH8(const std::vector<Triangle>& tri)
{
	triangles = tri;
	for (int i = 0; i < triangles.size(); i++)
	{
		deviceTriangles[i] = triangles[i].ToDevice();
	}
}

void BVH8::Init()
{
	// Fill the indices with integers starting from 0
	std::iota(triangleIdx.begin(), triangleIdx.end(), 0);
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

void BVH8::UpdateDeviceData()
{
	for (int i = 0; i < nodes.size(); i++)
	{
		deviceNodes[i] = nodes[i].ToDevice();
	}
	deviceTriangleIdx = triangleIdx;
}
