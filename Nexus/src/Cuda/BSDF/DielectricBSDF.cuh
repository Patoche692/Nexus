#pragma once

#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Cuda/Utils.cuh"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Random.cuh"
#include "Microfacet.cuh"
#include "Fresnel.cuh"

/* 
 *	Rough dielectric BSDF based on the paper "Microfacet Models for Refraction through Rough Surfaces"
 *	See https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
 */
struct DielectricBSDF
{
	float eta;
	float alpha;

	inline __device__ void PrepareBSDFData(const float3& wi,  const D_Material& material)
	{
		alpha = clamp((1.2f - 0.2f * sqrtf(fabs(wi.z))) * material.dielectric.roughness * material.dielectric.roughness, 1.0e-4f, 1.0f);
		eta = wi.z < 0.0f ? material.dielectric.ior : 1 / material.dielectric.ior;
	}

	inline __device__ bool Eval(const D_HitResult& hitResult, const float3& wi, const float3& wo, float3 throughput, float& pdf)
	{
		const float wiDotN = wi.z;
		const float woDotN = wo.z;

		const bool reflected = wiDotN * woDotN > 0.0f;

		float3 m;
		if (reflected)
			m = normalize(wo + wi);
		else
			m = -normalize(wi * eta + wo);

		float cosThetaT;
		const float wiDotM = dot(wi, m);
		const float woDotM = dot(wo, m);
		const float F = Fresnel::DieletricReflectance(1.0f / hitResult.material.dielectric.ior, wiDotM, cosThetaT);
		const float G = Microfacet::Smith_G2(alpha, woDotN, wiDotN);
		const float D = Microfacet::BeckmannD(alpha, m.z);

		if (reflected)
		{
			// BSDF times woDotN
			const float3 brdf = make_float3(F * G * D / (4.0f * fabs(wiDotN)));
			throughput = brdf;

			// pm * jacobian = pm * || dWhr / dWo ||
			pdf = F * D * m.z / (4.0f * fabs(wiDotM));
		}
		else
		{
			const float3 btdf = fabs(wiDotM * woDotM) * (1.0f - F) * G * D / (fabs(wiDotN * woDotN) * Square(eta * wiDotM + woDotM)) * hitResult.material.dielectric.albedo;
			throughput = btdf;

			pdf = (1.0f - F) * D * m.z * woDotM / Square(eta * wiDotM + woDotM);
		}

		return true;
	}

	inline __device__ bool Sample(const D_HitResult& hitResult, const float3& wi, float3& wo, float3& throughput, unsigned int& rngState)
	{
		const float3 m = Microfacet::SampleSpecularHalfBeckWalt(alpha, rngState);

		const float wiDotM = dot(wi, m);

		float cosThetaT;
		const float fr = Fresnel::DieletricReflectance(1.0f / hitResult.material.dielectric.ior, wiDotM, cosThetaT);

		// Randomly select a reflected or transmitted ray based on Fresnel reflectance
		if (Random::Rand(rngState) < fr)
		{
			// Specular
			wo = reflect(-wi, m);

			const float weight = Microfacet::WeightBeckmannWalter(alpha, abs(wiDotM), abs(wo.z), abs(wi.z), m.z);

			// Handle divisions by zero
			if (weight > 1.0e10)
				return false;

			// If the new ray is under the hemisphere, return
			if (wo.z * wi.z < 0.0f)
				return false;

			// We dont need to include the Fresnel term since it's already included when
			// we select between reflection and refraction (see paper page 7)
			throughput = make_float3(weight); // * F / fr
		}

		else
		{
			// Refraction
			wo = (eta * wiDotM - Utils::SgnE(wiDotM) * cosThetaT) * m - eta * wi;

			const float weight = Microfacet::WeightBeckmannWalter(alpha, abs(wiDotM), abs(wo.z), abs(wi.z), m.z);

			// Handle divisions by zero
			if (weight > 1.0e10)
				return false;

			if (wo.z * wi.z > 0.0f)
				return false;

			throughput = hitResult.material.dielectric.albedo * weight;
			// Same here, we don't need to include the Fresnel term
			//throughput = throughput * (1.0f - F) / (1.0f - fr)
		}
		return true;
	}
};