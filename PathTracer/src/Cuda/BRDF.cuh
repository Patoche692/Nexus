#pragma once

#include <cuda_runtime_api.h>
#include "Random.cuh"
#include "../Geometry/Materials/Material.h"

__device__ bool diffuseScatter(Material& material, float3& p, float3& attenuation, float3& normal, Ray& scattered, uint32_t& rngState)
{
	float3 scatterDirection = normal + Random::RandomUnitVector(rngState);
	scattered = Ray(p + normal * 0.001f, scatterDirection);
	attenuation *= material.diffuse.albedo;
	return true;
}
