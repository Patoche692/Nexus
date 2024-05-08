
#include "Cuda/Random.cuh"
#include "Utils/cuda_math.h"
#include "Geometry/Material.h"

struct ConductorBSDF
{

	inline __device__ bool Sample(const HitResult& hitResult, const float3& wi, float3& wo, float3& throughput, unsigned int& rngState)
	{

	}
};