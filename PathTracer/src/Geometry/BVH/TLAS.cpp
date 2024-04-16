#include "TLAS.h"

TLAS::TLAS(BVHInstance* bvhList, int N)
{
	blas = bvhList;
	blasCount = N;

	nodes = new TLASNode[2 * N];
	nodesIdx = new uint32_t[N];
	nodesUsed = 2;
}

void TLAS::Build()
{
	nodesUsed = 1;
	for (uint32_t i = 0; i < blasCount; i++)
	{
		nodesIdx[i] = nodesUsed;
		nodes[nodesUsed].aabbMin = blas[i].bounds.bMin;
		nodes[nodesUsed].aabbMax = blas[i].bounds.bMax;
		nodes[nodesUsed].blasIdx = i;
		nodes[nodesUsed++].leftRight = 0;
	}

	int nodeIndices = blasCount;
	int A = 0, B = FindBestMatch(nodeIndices, A);

	while (nodeIndices > 1)
	{
		int C = FindBestMatch(nodeIndices, B);
		if (A == C)
		{
			int nodeIdxA = nodesIdx[A], nodeIdxB = nodesIdx[B];
			TLASNode& nodeA = nodes[nodeIdxA];
			TLASNode& nodeB = nodes[nodeIdxB];
			TLASNode& newNode = nodes[nodesUsed];
			newNode.leftRight = nodeIdxA + (nodeIdxB << 16);
			newNode.aabbMin = fminf(nodeA.aabbMin, nodeB.aabbMin);
			newNode.aabbMax = fmaxf(nodeA.aabbMax, nodeB.aabbMax);
			nodesIdx[A] = nodesUsed++;
			nodesIdx[B] = nodesIdx[nodeIndices - 1];
			B = FindBestMatch(--nodeIndices, A);
		}
		else
			A = B, B = C;
	}
	nodes[0] = nodes[nodesIdx[A]];

}

int TLAS::FindBestMatch(int N, int A)
{
	float smallest = 1e30f;
	int bestB = -1;
	for (int B = 0; B < N; B++)
	{
		if (B != A)
		{
			float3 bMax = fmaxf(nodes[nodesIdx[A]].aabbMax, nodes[nodesIdx[B]].aabbMax);
			float3 bMin = fminf(nodes[nodesIdx[A]].aabbMin, nodes[nodesIdx[B]].aabbMin);
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
