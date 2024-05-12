#pragma once
#include "Utils/cuda_math.h"
#include "Ray.h"

struct Material {

	enum struct Type : char {
		DIFFUSE,
		CONDUCTOR,
		DIELECTRIC
	};

	//union {
	//	struct {
	//		float3 emission;
	//	} light;
	//	struct {
	//		float3 albedo;
	//	} diffuse;
	//	struct {
	//		float3 albedo;
	//		float roughness;
	//	} plastic;
	//	struct {
	//		float ior;
	//		float roughness;
	//	} dielectric;
	//};
	float3 diffuse;
	float ior;

	float3 emissive;
	float roughness;

	float transmittance;
	float opacity;

	int diffuseMapId = -1;
	//Type type;
};

struct HitResult
{
	float3 p;
	Ray rIn;
	float3 albedo;
	float3 normal;
	Material material;
};

