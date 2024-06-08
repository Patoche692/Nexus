#pragma once

#include "BVH8.cuh"
#include "Math/Mat4.h"
#include "Cuda/Geometry/AABB.cuh"

struct D_BVHInstance
{
	D_BVH8* bvh = nullptr;
	Mat4 invTransform;
	Mat4 transform;
	D_AABB bounds;
	int materialId;
};
