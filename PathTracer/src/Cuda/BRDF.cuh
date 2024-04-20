#pragma once

#include <cuda_runtime_api.h>
#include "Random.cuh"
#include "Geometry/Material.h"

inline __device__ bool diffuseScatter(HitResult& hitResult, float3& attenuation, Ray& scattered, uint32_t& rngState)
{
	float3 scatterDirection = hitResult.normal + Random::RandomUnitVector(rngState);
	scattered = Ray(hitResult.p, normalize(scatterDirection));
	attenuation = hitResult.albedo;
	return true;
}

inline __device__ bool plasticScattter(HitResult& hitResult, float3& attenuation, Ray& scattered, uint32_t& rngState)
{
	float3 reflected = reflect(normalize(hitResult.rIn.direction), hitResult.normal);
	scattered = Ray(hitResult.p, normalize(reflected + hitResult.material.plastic.roughness * Random::RandomUnitVector(rngState)));
	attenuation = hitResult.albedo;
	return dot(scattered.direction, hitResult.normal) > 0.0f;
}

inline __device__ float schlick(float cosine, float ri)
{
	float r0 = (1.0f - ri) / (1.0f + ri);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

inline __device__ bool refract(const float3& v, const float3& n, float niOverNt, float3& refracted)
{
	float3 uv = normalize(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - niOverNt * niOverNt * (1 - dt * dt);
	if (discriminant > 0.0f)
	{
		refracted = niOverNt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	// Cannot refract, then reflect
	return false;
}

inline __device__ bool dielectricScattter(HitResult& hitResult, float3& attenuation, Ray& scattered, uint32_t& rngState)
{
	float3 outwardNormal;
	float3 reflected = reflect(hitResult.rIn.direction, hitResult.normal);
	float niOverNt;
	float3 refracted;
	float reflectProb;
	float cosine;
	if (dot(hitResult.rIn.direction, hitResult.normal) > 0.0f)
	{
		outwardNormal = -hitResult.normal;
		niOverNt = hitResult.material.dielectric.ior;
		cosine = dot(hitResult.rIn.direction, hitResult.normal) / length(hitResult.rIn.direction);
		cosine = sqrt(1.0f - hitResult.material.dielectric.ior * hitResult.material.dielectric.ior * (1.0f - cosine * cosine));
	}
	else
	{
		outwardNormal = hitResult.normal;
		niOverNt = 1.0f / hitResult.material.dielectric.ior;
		cosine = -dot(hitResult.rIn.direction, hitResult.normal) / length(hitResult.rIn.direction);
	}
	if (refract(hitResult.rIn.direction, outwardNormal, niOverNt, refracted))
		//reflectProb = 0.0f;
		reflectProb = schlick(cosine, hitResult.material.dielectric.ior);
	else
		reflectProb = 1.0f;

	if (Random::Rand(rngState) < reflectProb)
	{
		scattered = Ray(hitResult.p, reflected);
	}
	else
	{
		scattered = Ray(hitResult.p, refracted);
	}
	return true;
}
