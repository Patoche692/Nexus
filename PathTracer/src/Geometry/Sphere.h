#pragma once
#include "Material.h"
#include "cuda/cuda_math.h"
#include "Hitable.h"

#define MAX_SPHERES 50

struct Sphere: Hitable
{
	__host__ __device__ Sphere(float radius, float3 position, Material material);
	__host__ __device__ virtual bool Hit(const Ray& r, float t_min, float t_max, HitResult& hitResult) const override;

	float radius;
	float3 position;
	Material material;
};

