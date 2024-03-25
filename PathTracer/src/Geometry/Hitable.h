#pragma once
#include "../Utils/cuda_math.h"
#include "Materials/Material.h"
#include "Ray.h"

struct HitResult
{
	float t;
	float3 normal;
	Material *material;
};

class Hitable
{
public:
	__host__ __device__ Hitable() { };
	__host__ __device__ virtual bool Hit(const Ray& r, float& t) const = 0;
};
