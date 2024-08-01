#pragma once
#include "Cuda/Random.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Scene/Material.cuh"
#include "Utils/Utils.h"

// Basic lambertian (diffuse) BSDF
struct D_LambertianBSDF
{
	inline __device__ bool Sample(const D_HitResult& hitResult, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
	{
		wo = Random::RandomCosineHemisphere(rngState);
		throughput = hitResult.material.diffuse.albedo;
		pdf = INV_PI * wo.z;
	}

	inline __device__ void PrepareBSDFData(const float3& wi, const D_Material& material)
	{

	}
};