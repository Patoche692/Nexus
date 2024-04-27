#pragma once
#include "Utils/cuda_math.h"
#include "Ray.h"

struct Material {

	enum struct Type : char {
		LIGHT,
		DIFFUSE,
		METAL,
		DIELECTRIC,
		CONDUCTOR
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
	float metalness;

	float3 emissive;
	float roughness;

	float transmissiveness;
	float reflectance;
	float opacity;
	float ior;

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

