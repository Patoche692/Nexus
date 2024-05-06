#include "Cuda/Random.cuh"
#include "Geometry/Ray.h"
#include "Geometry/Material.h"

struct LambertianBSDF
{
	float diffuse;

	inline __device__ bool Sample(HitResult& hitResult, float3& attenuation, float3& scattered, unsigned int& rngState)
	{
		
	}
};