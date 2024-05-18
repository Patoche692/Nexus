#pragma once
#include "Utils/cuda_math.h"
#include "Ray.h"

struct Material {

	enum struct Type : char {
		DIFFUSE,
		DIELECTRIC,
		CONDUCTOR
	};

	union {
		struct {
			float3 albedo;
		} diffuse;
		struct {
			float3 albedo;
			float roughness;
			float transmittance;
			float ior;
		} dielectric;
		struct {
			float3 ior;
			float3 k;
			float roughness;
		} conductor;
	};

	float3 emissive;
	float opacity;

	int diffuseMapId = -1;
	int emissiveMapId = -1;
	Type type;
};

struct HitResult
{
	float3 p;
	Ray rIn;
	float3 albedo;
	float3 normal;
	Material material;
};

