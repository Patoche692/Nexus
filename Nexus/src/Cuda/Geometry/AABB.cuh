#pragma once

#include <cuda_runtime_api.h>
#include "Ray.cuh"

struct D_AABB
{
	float3 bMin = make_float3(1e30f);
	float3 bMax = make_float3(-1e30f);

	static inline __device__ float IntersectionAABB(const D_Ray& ray, const float3& bMin, const float3& bMax)
	{
		float tx1 = (bMin.x - ray.origin.x) * ray.invDirection.x, tx2 = (bMax.x - ray.origin.x) * ray.invDirection.x;
		float tmin = fmin(tx1, tx2), tmax = fmax(tx1, tx2);
		float ty1 = (bMin.y - ray.origin.y) * ray.invDirection.y, ty2 = (bMax.y - ray.origin.y) * ray.invDirection.y;
		tmin = fmax(tmin, fmin(ty1, ty2)), tmax = fmin(tmax, fmax(ty1, ty2));
		float tz1 = (bMin.z - ray.origin.z) * ray.invDirection.z, tz2 = (bMax.z - ray.origin.z) * ray.invDirection.z;
		tmin = fmax(tmin, fmin(tz1, tz2)), tmax = fmin(tmax, fmax(tz1, tz2));
		if (tmax >= tmin && tmin < ray.hit.t && tmax > 0) return tmin; else return 1e30f;
	}
};