#pragma once
#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Geometry/Material.h"
#include "Random.cuh"

#define ONE_DIV_PI (0.31830988618f)

struct BSDF {

	// for 
	// lambertianeval => for diffuesd FUNCTION
	// lambertianPDF FUNCTION

	float3 specularF0;
	float3 diffuseReflectance;

	float roughness;
	float alpha;
	float alphaSquared;

	float3 F;

	float3 V;	// Opposite incoming direction
	float3 N;	// Normal
	float3 H;	// Half vector
	float3 L;	// Outgoing direction

	float NdotL;
	float NdotV;

	float LdotH;
	float NdotH;
	float VdotH;

	bool VbackFacing;
	bool LbachFacing;

	inline __device__ float4 GetRotationToZAxis(float3 direction)
	{
		if (direction.z < -0.99999f) return make_float4(1.0f, 0.0f, 0.0f, 0.0f);
		return normalize(make_float4(direction.y, -direction.x, 0.0f, 1.0f + direction.z));
	}

	inline __device__ float4 GetRotationFromZAxis(float3 direction)
	{
		if (direction.z < -0.99999f) return make_float4(1.0f, 0.0f, 0.0f, 0.0f);
		return normalize(make_float4(-direction.y, direction.x, 0.0f, 1.0f + direction.z));
	}

	inline __device__ float4 InvertRotation(float4 q)
	{
		return make_float4(-q.x, -q.y, -q.z, q.w);
	}

	inline __device__ float3 RotatePoint(float4 q, float3 v)
	{
		const float3 qAxis = make_float3(q.x, q.y, q.z);
		return 2.0f * dot(qAxis, v) * qAxis + (q.w * q.w - dot(qAxis, qAxis)) * v + 2.0f * q.w * cross(qAxis, v);
	}

	inline __device__ void PrepareBSDFData(float3& scatteredLocal, float3& Vlocal,  Material& material)
	{
		V = Vlocal;
		L = scatteredLocal;
		H = normalize(V + L);
		float3 N = make_float3(0.0f, 0.0f, 1.0f);

		NdotL = dot(N, L);
		NdotH = dot(N, H);
		NdotV = dot(N, V);
		VdotH = dot(V, H);
		LdotH = dot(L, H);

		diffuseReflectance = material.diffuse;
	}

	inline __device__ float lambertian()
	{
		return 1.0f;
	}

	inline __device__ float3 lambertianEval() {
		return diffuseReflectance * ONE_DIV_PI * NdotL;
	}

	inline __device__ bool Eval(HitResult& hitResult, float3& attenuation, float3& scattered, unsigned int& rngState)
	{
		float4 qRotationToZ = GetRotationToZAxis(hitResult.normal);
		float3 Vlocal = RotatePoint(qRotationToZ, -hitResult.rIn.direction);
		float3 Nlocal = make_float3(0.0f, 0.0f, 1.0f);

		float3 scatteredLocal = Random::RandomCosineHemisphere(rngState);

		PrepareBSDFData(scatteredLocal, Vlocal, hitResult.material);

		attenuation = diffuseReflectance * lambertian();

		scattered = normalize(RotatePoint(InvertRotation(qRotationToZ), scatteredLocal));

		return true;
	}
};