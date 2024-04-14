#pragma once
#include "Geometry/Mesh.h"
#include "Geometry/Ray.h"

#define BINS 8

struct BVHNode
{
	// Bounds
	float3 aabbMin, aabbMax;

	// Either the index of the left child node, or the first index in the triangles list
	union {
		uint32_t leftNode;
		uint32_t firstTriIdx;
	};

	// Number of triangles in the node (0 if not leaf)
	uint32_t triCount;

	__host__ __device__ bool IsLeaf() { return triCount > 0; }
	__host__ __device__ float Cost()
	{
		float3 diag = aabbMax - aabbMin;
		return (diag.x * diag.y + diag.y * diag.z + diag.x * diag.z) * triCount;
	}
};

class BVH
{
public:
	BVH() = default;
	BVH(std::vector<Triangle> triangles);

	void Build();

private:
	void Subdivide(uint32_t nodeIdx);
	void UpdateNodeBounds(uint32_t nodeIdx);
	float FindBestSplitPlane(BVHNode& node, int& axis, float& splitPos);

	Triangle* m_Triangles = nullptr;
	uint32_t* m_TriangleIdx = nullptr;
	uint32_t m_NodesUsed, m_TriCount;

public:
	BVHNode* nodes = nullptr;

	// Ray intersection (executed on the GPU)
	inline __host__ __device__ void Intersect(Ray& ray, uint32_t instanceIdx)
	{
		BVHNode* node = nodes, * stack[32];
		uint32_t stackPtr = 0;

		while (1)
		{
			if (node->IsLeaf())
			{
				for (uint32_t i = 0; i < node->triCount; i++)
					m_Triangles[i].Hit(ray, instanceIdx, m_TriangleIdx[i]);

				if (stackPtr == 0)
					break;
				else
					node = stack[--stackPtr];
				continue;
			}

			BVHNode* child1 = &nodes[node->leftNode];
			BVHNode* child2 = &nodes[node->leftNode + 1];
			float dist1 = AABB::intersectionAABB(ray, child1->aabbMin, child1->aabbMax);
			float dist2 = AABB::intersectionAABB(ray, child2->aabbMin, child2->aabbMax);

			if (dist1 > dist2)
			{
				Swap(dist1, dist2);
				Swap(child1, child2);
			}
			
			if (dist1 == 1e30f)
			{
				if (stackPtr == 0)
					break;
				else
					node = stack[--stackPtr];
			}
			else
			{
				node = child1;
				if (dist2 != 1e30f)
					stack[stackPtr++] = child2;
			}

		}
	}

	template<typename T>
	inline __host__ __device__ void Swap(T& a, T& b) {
		T c = a;
		a = b;
		b = c;
	}
};


