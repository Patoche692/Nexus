#pragma once
#include "Material.h"
#include "cuda/cuda_math.h"

#define MAX_SPHERES 50

struct Sphere
{
	float radius;
	float3 position;
	Material material;
};

