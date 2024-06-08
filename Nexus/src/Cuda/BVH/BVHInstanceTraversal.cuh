#pragma once

#include <cuda_runtime_api.h>
#include "BVHInstance.cuh"
#include "BVH8Traversal.cuh"

inline __device__ void IntersectBVHInstance(const D_BVHInstance& instance, D_Ray& ray, const uint32_t instanceIdx)
{
	D_Ray backupRay = ray;
	ray.origin = instance.invTransform.TransformPoint(ray.origin);
	ray.direction = instance.invTransform.TransformVector(ray.direction);
	ray.invDirection = 1 / ray.direction;

	IntersectBVH8(*instance.bvh, ray, instanceIdx);

	backupRay.hit = ray.hit;
	ray = backupRay;
}