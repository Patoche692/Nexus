#pragma once
#include "cuda/cuda_math.h"
#include "Material.h"
#include "Ray.h"

struct HitResult
{
	float t;
	float3 p;
	float3 normal;
	Material *material;
};

class Hitable
{
public:
	__host__ __device__ Hitable() { };
	__host__ __device__ virtual bool Hit(const Ray& r, HitResult& hitResult) const = 0;
};
