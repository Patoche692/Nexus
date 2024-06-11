#include "TLAS.h"

#include <cuda_runtime_api.h>

TLAS::TLAS(const std::vector<BVHInstance>& bvhList)
{
	blas = bvhList;
	deviceBlas = thrust::device_vector<D_BVHInstance>(blas.size());
}

void TLAS::Build()
{
	nodes.emplace_back();
	for (uint32_t i = 0; i < blas.size(); i++)
	{
		instancesIdx.push_back(i + 1);

		TLASNode node;
		node.aabbMin = blas[i].bounds.bMin;
		node.aabbMax = blas[i].bounds.bMax;
		node.blasIdx = i;
		node.leftRight = 0;
		nodes.push_back(node);
	}

	int nodeIndices = blas.size();
	int A = 0, B = FindBestMatch(nodeIndices, A);

	while (nodeIndices > 1)
	{
		int C = FindBestMatch(nodeIndices, B);
		if (A == C)
		{
			int nodeIdxA = instancesIdx[A], nodeIdxB = instancesIdx[B];
			TLASNode& nodeA = nodes[nodeIdxA];
			TLASNode& nodeB = nodes[nodeIdxB];
			TLASNode newNode;
			newNode.leftRight = nodeIdxA + (nodeIdxB << 16);
			newNode.aabbMin = fminf(nodeA.aabbMin, nodeB.aabbMin);
			newNode.aabbMax = fmaxf(nodeA.aabbMax, nodeB.aabbMax);
			nodes.push_back(newNode);
			instancesIdx[A] = nodes.size();
			instancesIdx[B] = instancesIdx[nodeIndices - 1];
			B = FindBestMatch(--nodeIndices, A);
		}
		else
			A = B, B = C;
	}
	nodes[0] = nodes[instancesIdx[A]];
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
	for (int i = 0; i < blas.size(); i++)
	{
		deviceBlas[i] = blas[i].ToDevice();
	}
	for (int i = 0; i < nodes.size(); i++)
	{
		deviceNodes[i] = nodes[i].ToDevice();
	}
}

D_TLAS TLAS::ToDevice()
{
	D_TLAS deviceTlas;
	deviceTlas.blas = thrust::raw_pointer_cast(deviceBlas.data());
	deviceTlas.nodes = thrust::raw_pointer_cast(deviceNodes.data());
	return deviceTlas;
}
