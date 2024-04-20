#pragma once
#include "Utils/cuda_math.h"
#include "Ray.h"
#include <OpenGL/Texture.h>

struct Material {

	enum struct Type : char {
		LIGHT,
		DIFFUSE,
		METAL,
		DIELECTRIC,
		CONDUCTOR
	};

	union {
		struct {
			float3 emission;
		} light;
		struct {
			float3 albedo;
			//uint32_t textureHandle; // for texture
			//Texture* texture;		https://choodle.portal.chalmers.se/laxM2Dt4x8sTXOpF/vote/EeG9hcycqeTpqYRN#edit
		} diffuse;
		struct {
			float3 albedo;
			float roughness;
		} plastic;
		struct {
			float ior;
			float roughness;
		} dielectric;
	};

	Type type;
};

struct HitResult
{
	float3 p;
	Ray rIn;
	float3 normal;
	Material material;
};

