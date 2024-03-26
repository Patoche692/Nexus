#pragma once

#include <cuda_runtime_api.h>


enum struct CMaterialType : char
{
	LIGHT,
	DIFFUSE,
	PLASTIC,
	DIELECTRIC,
	CONDUCTOR
};

union CMaterial {
	struct {
		float3 albedo;
	} diffuse;
};