#pragma once
#include "cuda/cuda_math.h"

// Make it a struct so that its properties are publicly available to CUDA without setters / getters

struct Ray
{
	__host__ __device__ Ray() = default;
	__host__ __device__ Ray(float3 o, float3 d)
		:origin(o), direction(d) {};

	inline __host__ __device__ float3 PointAtParameter(float t) const { return origin + direction * t; };

	float3 origin;
	float3 direction;
};
