#include "Cuda/Random.cuh"
#include "Geometry/Ray.h"
#include "Geometry/Material.h"

// Basic lambertian (diffuse) BSDF
struct LambertianBSDF
{
	inline __device__ bool Sample(const HitResult& hitResult, const float3& wi, float3& wo, float3& throughput, unsigned int& rngState)
	{
		wo = Random::RandomCosineHemisphere(rngState);
		throughput = hitResult.material.diffuse;
	}
};