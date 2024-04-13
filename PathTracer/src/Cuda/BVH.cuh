#pragma once
#include "Geometry/Mesh.h"
#include "Geometry/Ray.h"


class BVH
{
public:
	BVH() = default;
	BVH(Mesh& mesh);

	void Build();

	inline __host__ __device__ void Intersect(Ray& ray);

private:
	void Subdivide();
	void UpdateNodeBounds();
	float FindBestSplitPlane();

};

