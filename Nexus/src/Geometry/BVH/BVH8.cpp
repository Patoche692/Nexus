#include "BVH8.h"
#include <numeric>

BVH8::BVH8(const std::vector<Triangle>& tri)
{
	triangles = tri;
	deviceTriangles = DeviceVector<Triangle, D_Triangle>(tri);
}

void BVH8::Init()
{
	// Fill the indices with integers starting from 0
	std::iota(triangleIdx.begin(), triangleIdx.end(), 0);
}

D_BVH8 BVH8::ToDevice(const BVH8& bvh)
{
	D_BVH8 deviceBvh;
	deviceBvh.triangles = bvh.deviceTriangles.Data();
	deviceBvh.triangleIdx = bvh.deviceTriangleIdx.Data();
	deviceBvh.nodes = bvh.deviceNodes.Data();
	deviceBvh.nodesUsed = bvh.nodes.size();
	deviceBvh.triCount = bvh.triangles.size();
	return deviceBvh;
}

void BVH8::UpdateDeviceData()
{
	for (int i = 0; i < nodes.size(); i++)
	{
		deviceNodes[i] = nodes[i];
	}
	deviceTriangleIdx = triangleIdx;
}
