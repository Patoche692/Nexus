#pragma once
#include "Geometry/Mesh.h"
#include "Geometry/Ray.h"

struct BVHNode
{
	// Bounds
	float3 aabbMin, aabbMax;

	// Either the first index of the child node, or the first index in the triangles list
	uint32_t leftFirst;

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
	BVH(Mesh& mesh);

	void Build();

	inline __host__ __device__ void Intersect(Ray& ray)
	{

	}

private:
	void Subdivide();
	void UpdateNodeBounds();
	float FindBestSplitPlane();

};

