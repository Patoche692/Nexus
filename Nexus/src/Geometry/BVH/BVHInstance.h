#pragma once

#include <cuda_runtime_api.h>

#include "BVH.h"
#include "BVH8.h"
#include "Math/Mat4.h"
#include "Geometry/AABB.h"

class BVHInstance
{
public:
	BVHInstance() = default;
	BVHInstance(BVH8* blas) : bvh(blas) {
		Mat4 m;
		SetTransform(m); 
	}

	void SetTransform(Mat4& t);
	void SetTransform(float3 pos, float3 r, float3 s);

	void AssignMaterial(int mIdx);

public:
	BVH8* bvh = nullptr;
	Mat4 invTransform;
	Mat4 transform;
	AABB bounds;
	int materialId;

	//inline __host__ __device__ void Intersect(Ray& ray, uint32_t instanceIdx)
	//{
	//	Ray backupRay = ray;
	//	ray.origin = invTransform.TransformPoint(ray.origin);
	//	ray.direction = invTransform.TransformVector(ray.direction);
	//	ray.invDirection = 1 / ray.direction;

	//	bvh->Intersect(ray, instanceIdx);

	//	backupRay.hit = ray.hit;
	//	ray = backupRay;
	//}
};
