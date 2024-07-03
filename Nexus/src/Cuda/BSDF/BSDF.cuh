#pragma once

#include <cuda_runtime_api.h>
#include "Cuda/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"

struct BSDF {

	template<typename T>
	inline __device__ static bool Sample(const D_HitResult& hitResult, const float3& wi, float3& wo, float3& throughput, unsigned int& rngState)
	{
		T bsdf;
		bsdf.PrepareBSDFData(wi, hitResult.material);
		return bsdf.Sample(hitResult, wi, wo, throughput, rngState);
	}
};