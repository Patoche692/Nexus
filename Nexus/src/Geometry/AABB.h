#pragma once
#include "Utils/cuda_math.h"
#include "Ray.h"

struct AABB
{
	AABB() = default;
	AABB(float3 min, float3 max) : bMin(min), bMax(max) { }

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

	float3 bMin = make_float3(1e30f);
	float3 bMax = make_float3(-1e30f);
};

