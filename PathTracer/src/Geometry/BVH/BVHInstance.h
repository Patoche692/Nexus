#pragma once

#include <cuda_runtime_api.h>

#include "BVH.h"
#include "Math/Mat4.h"
#include "Geometry/AABB.h"

class BVHInstance
{
public:
	BVHInstance() = default;
	BVHInstance(BVH* blas) : m_Bvh(blas) {
		Mat4 m;
		SetTransform(m); 
	}
	void SetTransform(Mat4& transform);

private:
	BVH* m_Bvh = nullptr;
	Mat4 m_InvTransform;
public:
	AABB bounds;

	inline __host__ __device__ void Intersect(Ray& ray, uint32_t instanceIdx)
	{
		Ray backupRay = ray;
		ray.origin = m_InvTransform.TransformPoint(ray.origin);
		ray.direction = m_InvTransform.TransformVector(ray.direction);
		ray.invDirection = 1 / ray.direction;

		m_Bvh->Intersect(ray, instanceIdx);

		backupRay.hit = ray.hit;
		ray = backupRay;
	}
};
