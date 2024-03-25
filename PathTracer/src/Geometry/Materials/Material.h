#pragma once
#include "../../Utils/cuda_math.h"
#include "../Ray.h"

struct Material {
	__host__  __device__ Material() {};

	virtual inline __host__ __device__ bool Scatter(float3& p, float3& attenuation, float3& normal, Ray& scattered, uint32_t& rngState) = 0;
	virtual __host__ __device__ size_t GetSize() { return sizeof(*this); };

	uint32_t id;
};

