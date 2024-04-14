#pragma once

#include "BVH.h"
#include "Math/Mat4.h"
#include "Geometry/AABB.h"

class BVHInstance
{
public:
	BVHInstance() = default;
	BVHInstance(BVH* blas) : bvh(blas) {
		Mat4 m;
		SetTransform(m); 
	}
	void SetTransform(Mat4& transform);
	void Intersect(Ray& ray);

private:
	BVH* bvh = nullptr;
	Mat4 invTransform;
public:
	AABB bounds;
};
