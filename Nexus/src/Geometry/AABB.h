#pragma once
#include "Utils/cuda_math.h"
#include "Ray.h"

struct AABB
{
	float3 bMin = make_float3(1e30f);
	float3 bMax = make_float3(-1e30f);

	void Grow(float3 p)
	{
		bMin = fminf(bMin, p);
		bMax = fmaxf(bMax, p);
	}

	void Grow(AABB& other)
	{
		if (other.bMin.x != 1e30f)
		{
			Grow(other.bMin);
			Grow(other.bMax);
		}
	}

	// Return the area of three faces (thus the actual aabb area divided by 2,
	// it avoids an unnecessary multiplication for the surface area heuristic)
	float Area()
	{
		float3 diff = bMax - bMin;
		return diff.x * diff.y + diff.y * diff.z + diff.x * diff.z;
	}

	__host__ __device__ static inline float intersectionAABB(const Ray& ray, const float3& bMin, const float3& bMax)
	{
		float tx1 = (bMin.x - ray.origin.x) * ray.invDirection.x, tx2 = (bMax.x - ray.origin.x) * ray.invDirection.x;
		float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
		float ty1 = (bMin.y - ray.origin.y) * ray.invDirection.y, ty2 = (bMax.y - ray.origin.y) * ray.invDirection.y;
		tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
		float tz1 = (bMin.z - ray.origin.z) * ray.invDirection.z, tz2 = (bMax.z - ray.origin.z) * ray.invDirection.z;
		tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
		if (tmax >= tmin && tmin < ray.hit.t && tmax > 0) return tmin; else return 1e30f;
	}
};

