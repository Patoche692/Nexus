#pragma once
#include <string>
#include "Cuda/Material.cuh"
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
	float intensity;
	float opacity;

	int diffuseMapId = -1;
	int emissiveMapId = -1;
	Type type;

	D_Material ToDevice() const
	{
		D_Material deviceMaterial;
		memcpy(&deviceMaterial, this, sizeof(D_Material));
		return deviceMaterial;
	}

	static std::string GetMaterialTypesString()
	{
		std::string materialTypes;
		materialTypes.append("Diffuse");
		materialTypes.push_back('\0');
		materialTypes.append("Dielectric");
		materialTypes.push_back('\0');
		materialTypes.append("Conductor");
		materialTypes.push_back('\0');
		return materialTypes;
	}
};

