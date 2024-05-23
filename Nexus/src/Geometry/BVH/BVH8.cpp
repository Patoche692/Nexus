#include "BVH8.h"

BVH8::BVH8(BVH* bvh2)
{
	triangles = bvh2->triangles;
	triangleIdx = bvh2->triangleIdx;
	triCount = bvh2->triCount;
	// TODO: How many nodes should be allocated?
	nodes = new BVH8Node[triCount * 8];
	nodesUsed = 1;
}

BVH8::~BVH8()
{
	delete[] nodes;
}
