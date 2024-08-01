#pragma once

#include <cuda_runtime_api.h>
#include "BVHInstance.cuh"
#include "BVH8Traversal.cuh"

inline __device__ void BVHInstanceTrace(const D_BVHInstance& instance, const D_BVH8* bvhs, D_Ray& ray, const uint32_t instanceIdx)
{
	D_Ray backupRay = ray;
	ray.origin = instance.invTransform.TransformPoint(ray.origin);
	ray.direction = instance.invTransform.TransformVector(ray.direction);
	ray.invDirection = 1.0f / ray.direction;

	BVH8Trace(bvhs[instance.bvhIdx], ray, instanceIdx);

	backupRay.hit = ray.hit;
	ray = backupRay;
}

inline __device__ bool BVHInstanceTraceShadow(const D_BVHInstance& instance, const D_BVH8* bvhs, D_Ray& ray)
{
	D_Ray backupRay = ray;
	ray.origin = instance.invTransform.TransformPoint(ray.origin);
	ray.direction = instance.invTransform.TransformVector(ray.direction);
	ray.invDirection = 1.0f / ray.direction;

	bool anyHit = BVH8TraceShadow(bvhs[instance.bvhIdx], ray);

	ray = backupRay;

	return anyHit;
}
