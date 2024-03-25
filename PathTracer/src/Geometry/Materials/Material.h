#pragma once
#include "../../Utils/cuda_math.h"

struct Material {
	__host__  __device__ Material() {};
	__host__ __device__ Material(float3 c) : color(c) {};
	//virtual inline __host__ __device__ bool Scatter();
	float3 color;
	uint32_t id;
};

