#pragma once
#include "Utils/cuda_math.h"
#include "Medium.h"

struct Ray
{
	__host__ __device__ Ray() = default;
	__host__ __device__ Ray(float3 o, float3 d)
		:origin(o), direction(d) {};

	// Ray origin
	float3 origin = make_float3(0.0f);

	// Ray direction
	float3 direction = make_float3(0.0f, 0.0f, 1.0f);

	// Ray inverse direction (reduce divisions for optimization)
	float3 invDirection = make_float3(0.0f, 0.0f, 1.0f);

	// Ray hit distance
	float t = 0.0f;

	// Medium
	Medium medium = { 1.0f };

	inline __host__ __device__ float3 PointAtParameter(float t) const { return origin + direction * t; };
};
