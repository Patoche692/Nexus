#include "TLAS.h"

#include <cuda_runtime_api.h>

TLAS::TLAS(const std::vector<BVHInstance>& instancesList, const std::vector<BVH8>& bvhList)
{
	bvhInstances = instancesList;
	bvhs = bvhList;
}

void TLAS::Build()
{
	nodes.clear();
	instancesIdx.clear();

	nodes.emplace_back();

	for (uint32_t i = 0; i < bvhInstances.size(); i++)
	{
		instancesIdx.push_back(i + 1);

		TLASNode node;
		node.aabbMin = bvhInstances[i].GetBounds().bMin;
		node.aabbMax = bvhInstances[i].GetBounds().bMax;
		node.blasIdx = i;
		node.left = 0;
		node.right = 0;
		nodes.push_back(node);
	}

	int nodeIndices = bvhInstances.size();
	int A = 0, B = FindBestMatch(nodeIndices, A);

	// TODO: handle case B = -1 (only one instance)
	while (nodeIndices > 1)
	{
		int C = FindBestMatch(nodeIndices, B);
		if (A == C)
		{
			int nodeIdxA = instancesIdx[A], nodeIdxB = instancesIdx[B];
			TLASNode& nodeA = nodes[nodeIdxA];
			TLASNode& nodeB = nodes[nodeIdxB];
			const uint32_t blasCountA = nodeA.IsLeaf() ? 1 : nodeA.blasCount;
			const uint32_t blasCountB = nodeB.IsLeaf() ? 1 : nodeB.blasCount;

			TLASNode newNode;
			newNode.left = nodeIdxB;
			newNode.right = nodeIdxA;
			newNode.blasCount = blasCountA + blasCountB;
			newNode.aabbMin = fminf(nodeA.aabbMin, nodeB.aabbMin);
			newNode.aabbMax = fmaxf(nodeA.aabbMax, nodeB.aabbMax);
			instancesIdx[A] = nodes.size();
			instancesIdx[B] = instancesIdx[nodeIndices - 1];
			nodes.push_back(newNode);
			B = FindBestMatch(--nodeIndices, A);
		}
		else
			A = B, B = C;
	}

	nodes[0] = nodes[instancesIdx[A]];
	TLASNode& nodeA = nodes[nodes[0].left];
	TLASNode& nodeB = nodes[nodes[0].right];
	const uint32_t blasCountA = nodeA.IsLeaf() ? 1 : nodeA.blasCount;
	const uint32_t blasCountB = nodeB.IsLeaf() ? 1 : nodeB.blasCount;
	nodes[0].blasCount = blasCountA + blasCountB;

	deviceNodes = DeviceVector<TLASNode, D_TLASNode>(nodes.size());
}

int TLAS::FindBestMatch(int N, int A)
{
	float smallest = 1e30f;
	int bestB = -1;
	for (int B = 0; B < N; B++)
	{
		if (B != A)
		{
			float3 bMax = fmaxf(nodes[instancesIdx[A]].aabbMax, nodes[instancesIdx[B]].aabbMax);
			float3 bMin = fminf(nodes[instancesIdx[A]].aabbMin, nodes[instancesIdx[B]].aabbMin);
			float3 e = bMax - bMin;
			float surfaceArea = e.x * e.y + e.y * e.z + e.x * e.z;
			if (surfaceArea < smallest)
			{
				smallest = surfaceArea;
				bestB = B;
			}
		}

	}
	return bestB;
}

void TLAS::UpdateDeviceData()
{
	deviceBlas = DeviceVector<BVHInstance, D_BVHInstance>(bvhInstances);
	deviceNodes = DeviceVector<TLASNode, D_TLASNode>(nodes);
	deviceBvhs = DeviceVector<BVH8, D_BVH8>(bvhs);
}

D_TLAS TLAS::ToDevice(const TLAS& tlas)
{
	D_TLAS deviceTlas;
	deviceTlas.blas = tlas.deviceBlas.Data();
	deviceTlas.nodes = tlas.deviceNodes.Data();
	deviceTlas.bvhs = tlas.deviceBvhs.Data();
	deviceTlas.instanceCount = tlas.bvhInstances.size();
	return deviceTlas;
}
