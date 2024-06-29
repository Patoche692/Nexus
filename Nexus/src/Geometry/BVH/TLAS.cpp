#include "TLAS.h"

#include <cuda_runtime_api.h>

TLAS::TLAS(const std::vector<BVHInstance>& bvhList, const std::vector<BVH8>& bvhs)
{
	m_Blas = bvhList;
	m_Bvhs = bvhs;
	m_DeviceBlas = DeviceVector<BVHInstance, D_BVHInstance>(m_Blas.size());
	m_DeviceBvhs = DeviceVector<BVH8, D_BVH8>(bvhs);
}

void TLAS::Build()
{
	m_Nodes.emplace_back();
	for (uint32_t i = 0; i < m_Blas.size(); i++)
	{
		m_InstancesIdx.push_back(i + 1);

		TLASNode node;
		node.aabbMin = m_Blas[i].GetBounds().bMin;
		node.aabbMax = m_Blas[i].GetBounds().bMax;
		node.blasIdx = i;
		node.leftRight = 0;
		m_Nodes.push_back(node);
	}

	int nodeIndices = m_Blas.size();
	int A = 0, B = FindBestMatch(nodeIndices, A);

	while (nodeIndices > 1)
	{
		int C = FindBestMatch(nodeIndices, B);
		if (A == C)
		{
			int nodeIdxA = m_InstancesIdx[A], nodeIdxB = m_InstancesIdx[B];
			TLASNode& nodeA = m_Nodes[nodeIdxA];
			TLASNode& nodeB = m_Nodes[nodeIdxB];
			TLASNode newNode;
			newNode.leftRight = nodeIdxA + (nodeIdxB << 16);
			newNode.aabbMin = fminf(nodeA.aabbMin, nodeB.aabbMin);
			newNode.aabbMax = fmaxf(nodeA.aabbMax, nodeB.aabbMax);
			m_Nodes.push_back(newNode);
			m_InstancesIdx[A] = m_Nodes.size();
			m_InstancesIdx[B] = m_InstancesIdx[nodeIndices - 1];
			B = FindBestMatch(--nodeIndices, A);
		}
		else
			A = B, B = C;
	}
	m_Nodes[0] = m_Nodes[m_InstancesIdx[A]];
}

int TLAS::FindBestMatch(int N, int A)
{
	float smallest = 1e30f;
	int bestB = -1;
	for (int B = 0; B < N; B++)
	{
		if (B != A)
		{
			float3 bMax = fmaxf(m_Nodes[m_InstancesIdx[A]].aabbMax, m_Nodes[m_InstancesIdx[B]].aabbMax);
			float3 bMin = fminf(m_Nodes[m_InstancesIdx[A]].aabbMin, m_Nodes[m_InstancesIdx[B]].aabbMin);
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
	for (int i = 0; i < m_Blas.size(); i++)
	{
		m_DeviceBlas[i] = m_Blas[i];
	}
	for (int i = 0; i < m_Nodes.size(); i++)
	{
		m_DeviceNodes[i] = m_Nodes[i];
	}
}

D_TLAS TLAS::ToDevice(const TLAS& tlas)
{
	D_TLAS deviceTlas;
	deviceTlas.blas = tlas.m_DeviceBlas.Data();
	deviceTlas.nodes = tlas.m_DeviceNodes.Data();
	deviceTlas.bvhs = tlas.m_DeviceBvhs.Data();
	return deviceTlas;
}
