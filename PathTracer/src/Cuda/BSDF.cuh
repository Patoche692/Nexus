#pragma once
#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#define ONE_DIV_PI (0.31830988618f)

struct BSDF {

	// for 
	// lambertianeval => for diffuesd FUNCTION
	// lambertianPDF FUNCTION

	// material properties: 
	float3 diffuseReflectance;
	float3 specular;

	float NdotL;


	inline __device__ float lambertian(const BSDF data)
	{
		return 1.0f;
	}

	inline __device__ float3 lambertianEval() {
		return ONE_DIV_PI * NdotL * diffuseReflectance;
	}

	inline __device__ float3 lambertianPDF(const BSDF data) {
		
	}

	inline __device__ float diffusePdf(const BSDF data) {
		return data.NdotL * ONE_DIV_PI;
	}
};