#include <vector>
#include "BVH.h"
#include "Geometry/AABB.h"
#include "Utils/Utils.h"

BVH::BVH(std::vector<Triangle>& tri)
{
	triCount = tri.size();
	triangles = new Triangle[triCount];
	nodes = new BVHNode[2 * triCount];
	triangleIdx = new uint32_t[triCount];

	memcpy(triangles, tri.data(), triCount * sizeof(Triangle));

	Build();
}

BVH::~BVH()
{
	delete[] triangles;
	delete[] nodes;
	delete[] triangleIdx;
}

void BVH::Build()
{
	nodesUsed = 2;

	for (uint32_t i = 0; i < triCount; i++)
		triangleIdx[i] = i;

	BVHNode& root = nodes[0];
	root.leftNode = 0;
	root.triCount = triCount;

	UpdateNodeBounds(0);
	Subdivide(0);
}

void BVH::Subdivide(uint32_t nodeIdx)
{
	BVHNode& node = nodes[nodeIdx];
	int axis;
	float splitPos;
	float splitCost = FindBestSplitPlane(node, axis, splitPos);
	float nodeCost = node.Cost();

	if (splitCost > nodeCost)
		return;

	int i = node.firstTriIdx;
	int j = i + node.triCount - 1;

	while (i <= j)
	{
		float centroidCoord = *((float*)&triangles[triangleIdx[i]].centroid + axis);
		if (centroidCoord < splitPos)
			i++;
		else
			Utils::Swap(triangleIdx[i], triangleIdx[j--]);
	}
	
	int leftCount = i - node.firstTriIdx;
	if (leftCount == 0 || leftCount == node.triCount)
		return;

	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;

	nodes[leftChildIdx].firstTriIdx = node.firstTriIdx;
	nodes[leftChildIdx].triCount = leftCount;
	nodes[rightChildIdx].firstTriIdx = i;
	nodes[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftNode = leftChildIdx;
	node.triCount = 0;

	UpdateNodeBounds(leftChildIdx);
	UpdateNodeBounds(rightChildIdx);

	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}

void BVH::UpdateNodeBounds(uint32_t nodeIdx)
{
	BVHNode& node = nodes[nodeIdx];
	node.aabbMin = make_float3(1e30f);
	node.aabbMax = make_float3(-1e30f);
	for (uint32_t first = node.firstTriIdx, i = 0; i < node.triCount; i++)
	{
		uint32_t leafTriIdx = triangleIdx[first + i];
		Triangle& leafTri = triangles[leafTriIdx];
		node.aabbMin = fminf(node.aabbMin, leafTri.pos0);
		node.aabbMin = fminf(node.aabbMin, leafTri.pos1);
		node.aabbMin = fminf(node.aabbMin, leafTri.pos2);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.pos0);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.pos1);
		node.aabbMax = fmaxf(node.aabbMax, leafTri.pos2);
	}
}

float BVH::FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos)
{
	float bestCost = 1e30f;
	for (int a = 0; a < 3; a++)
	{
		float boundsMin = 1e30f, boundsMax = -1e30f;
		for (uint32_t i = 0; i < node.triCount; i++)
		{
			Triangle& triangle = triangles[triangleIdx[node.firstTriIdx + i]];
			boundsMin = fmin(boundsMin, *((float*)&triangle.centroid + a));
			boundsMax = fmax(boundsMax, *((float*)&triangle.centroid + a));
		}
		if (boundsMin == boundsMax)
			continue;

		struct Bin { AABB bounds; int triCount = 0; } bins[BINS];
		float scale = BINS / (boundsMax - boundsMin);

		for (uint32_t i = 0; i < node.triCount; i++)
		{
			Triangle& triangle = triangles[triangleIdx[node.firstTriIdx + i]];
			float centroidCoord = *((float*)&triangle.centroid + a);
			int binIdx = min((int)(BINS - 1), (int)((centroidCoord - boundsMin) * scale));
			bins[binIdx].triCount++;
			bins[binIdx].bounds.Grow(triangle.pos0);
			bins[binIdx].bounds.Grow(triangle.pos1);
			bins[binIdx].bounds.Grow(triangle.pos2);
		}

		float leftArea[BINS - 1], rightArea[BINS - 1];
		int leftCount[BINS - 1], rightCount[BINS - 1];
		AABB leftBox, rightBox;
		int leftSum = 0, rightSum = 0;

		for (int i = 0; i < BINS - 1; i++)
		{
			leftSum += bins[i].triCount;
			leftCount[i] = leftSum;
			leftBox.Grow(bins[i].bounds);
			leftArea[i] = leftBox.Area();

			rightSum += bins[BINS - 1 - i].triCount;
			rightCount[BINS - 2 - i] = rightSum;
			rightBox.Grow(bins[BINS - 1 - i].bounds);
			rightArea[BINS - 2 - i] = rightBox.Area();
		}

		scale = (boundsMax - boundsMin) / BINS;
		for (int i = 0; i < BINS - 1; i++)
		{
			float planeCost = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
			if (planeCost < bestCost)
			{
				axis = a;
				splitPos = boundsMin + scale * (i + 1);
				bestCost = planeCost;
			}
		}
	}
	return bestCost;
}


