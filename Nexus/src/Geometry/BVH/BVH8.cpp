#include "BVH8.h"
#include <numeric>

BVH8::BVH8(const std::vector<Triangle>& tri)
{
	triangles = tri;
	triangleIdx = std::vector<uint32_t>(triangles.size());
}

void BVH8::Init()
{
	// Fill the indices with integers starting from 0
	std::iota(triangleIdx.begin(), triangleIdx.end(), 0);
}

