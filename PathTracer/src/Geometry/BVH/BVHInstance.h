#pragma once

#include <cuda_runtime_api.h>

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
	void SetTransform(Mat4& t);
	void SetTransform(float3 pos, float3 r, float3 s);

	void SetPosition(float3 pos);

	// Rotation is in degrees
	void SetRotation(float3 axis, float angle);
	void SetRotationX(float angle);
	void SetRotationY(float angle);
	void SetRotationZ(float angle);

	void SetScale(float s);
	void SetScale(float3 s);

	void AssignMaterial(int mIdx);

public:
	BVH* bvh = nullptr;
	Mat4 invTransform;
	Mat4 transform;
	float3 position = make_float3(0.0f);
	float3 rotation = make_float3(0.0f);
	float3 scale = make_float3(1.0f);
	AABB bounds;
	int materialId;

	inline __host__ __device__ void Intersect(Ray& ray, uint32_t instanceIdx)
	{
		Ray backupRay = ray;
		ray.origin = invTransform.TransformPoint(ray.origin);
		ray.direction = invTransform.TransformVector(ray.direction);
		ray.invDirection = 1 / ray.direction;

		bvh->Intersect(ray, instanceIdx);

		backupRay.hit = ray.hit;
		ray = backupRay;
	}
};
