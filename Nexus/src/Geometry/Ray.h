#pragma once
#include "Utils/cuda_math.h"
#include "Medium.h"


struct Intersection
{
	// Ray hit distance
	float t = 1.0e30f;
	// Barycentric coordinates;
	float u, v;

	uint32_t triIdx = 0;
	uint32_t instanceIdx = 0;
};

struct Ray
{
	__host__ __device__ Ray() = default;
	__host__ __device__ Ray(float3 o, float3 d)
		:origin(o), direction(d), invDirection(1/direction) {};

	// Ray origin
	float3 origin = make_float3(0.0f);

	// Ray direction
	float3 direction = make_float3(0.0f, 0.0f, 1.0f);

	// Ray inverse direction (reduce divisions for optimization)
	float3 invDirection = make_float3(0.0f, 0.0f, 1.0f);

	Intersection hit;

	inline __host__ __device__ float3 PointAtParameter(float t) const { return origin + direction * t; };
};


