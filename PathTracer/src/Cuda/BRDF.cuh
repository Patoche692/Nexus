#pragma once

#include <cuda_runtime_api.h>
#include "Random.cuh"
#include "../Geometry/Materials/Material.h"

inline __device__ bool diffuseScatter(HitResult& hitResult, float3& attenuation, Ray& scattered, uint32_t& rngState)
{
	float3 scatterDirection = hitResult.normal + Random::RandomUnitVector(rngState);
	scattered = Ray(hitResult.p + hitResult.normal * 0.001f, scatterDirection);
	attenuation *= hitResult.material.diffuse.albedo;
	return true;
}

inline __device__ bool plasticScattter(HitResult& hitResult, float3& attenuation, Ray& scattered, uint32_t& rngState)
{
	float3 reflected = reflect(normalize(hitResult.rIn.direction), hitResult.normal);
	scattered = Ray(hitResult.p + hitResult.normal * 0.001f, reflected + hitResult.material.plastic.roughness * Random::RandomUnitVector(rngState));
	attenuation *= hitResult.material.diffuse.albedo;
	return dot(scattered.direction, hitResult.normal) > 0.0f;
}
