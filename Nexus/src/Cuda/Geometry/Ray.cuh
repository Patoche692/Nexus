#pragma once

#include "Utils/Utils.h"
#include "Cuda/Material.cuh"

struct D_Intersection
{
	// Ray hit distance
	float t = 1.0e30f;
	// Barycentric coordinates;
	float u, v;

	uint32_t triIdx = -1;
	uint32_t instanceIdx = -1;
};

struct D_Ray
{
	D_Ray() = default;
	D_Ray(float3 o, float3 d)
		:origin(o), direction(d), invDirection(1/direction) {};

	// Ray origin
	float3 origin = make_float3(0.0f);

	// Ray direction
	float3 direction = make_float3(0.0f, 0.0f, 1.0f);

	// Ray inverse direction (reduce divisions for optimization)
	float3 invDirection = make_float3(0.0f, 0.0f, 1.0f);

	D_Intersection hit;

	inline  __device__ float3 PointAtParameter(float t) const { return origin + direction * t; };
};

struct D_HitResult
{
	float3 p;
	D_Ray rIn;
	float3 albedo;
	float3 normal;
	D_Material material;
};
