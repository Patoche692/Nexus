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
	virtual inline __host__ __device__ bool Hit(const Ray& r, float& t) = 0;
};
