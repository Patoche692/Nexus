#pragma once

#include <cuda_runtime_api.h>


enum struct MaterialType : char
{
	LIGHT,
	DIFFUSE,
	PLASTIC,
	DIELECTRIC,
	CONDUCTOR
};

//union Material {
//	struct {
//		float3 albedo;
//	} diffuse;
//};

//__device__ Material* materials;