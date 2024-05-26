#pragma once

#include "Geometry/BVH/BVHInstance.h"
#include "BVH2.cuh"
#include "BVH8.cuh"

inline __device__ void IntersectBVHInstance(const BVHInstance& instance, Ray& ray, const uint32_t instanceIdx)
{
	Ray backupRay = ray;
	ray.origin = instance.invTransform.TransformPoint(ray.origin);
	ray.direction = instance.invTransform.TransformVector(ray.direction);
	ray.invDirection = 1 / ray.direction;

	IntersectBVH8(*instance.bvh, ray, instanceIdx);

	backupRay.hit = ray.hit;
	ray = backupRay;
}
