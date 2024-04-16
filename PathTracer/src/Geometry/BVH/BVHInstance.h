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

	void Translate(float3 pos);

	// Rotation is in degrees
	void Rotate(float3 axis, float angle);
	void RotateX(float angle);
	void RotateY(float angle);
	void RotateZ(float angle);
	void Scale(float scale);
	void Scale(float3 scale);

	void AssignMaterial(int mIdx);

public:
	BVH* bvh = nullptr;
	Mat4 invTransform;
	Mat4 transform;
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

// A BVH instance can be considered as a Mesh instance
typedef BVHInstance MeshInstance;
