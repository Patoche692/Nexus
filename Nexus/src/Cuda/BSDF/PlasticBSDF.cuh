#pragma once

#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Cuda/Scene/Material.cuh"
#include "Cuda/Geometry/Ray.cuh"
#include "Cuda/Random.cuh"
#include "Microfacet.cuh"
#include "Fresnel.cuh"

/* 
 * Rough plastic BSDF. It uses a basic non physically accurate microfacet model (specular + diffuse)
 * It would be great to implement a layered model for this material, as described in
 * "Arbitrarily Layered Micro-Facet Surfaces" by Weidlich and Wilkie
 * See https://www.cg.tuwien.ac.at/research/publications/2007/weidlich_2007_almfs/weidlich_2007_almfs-paper.pdf
 */
struct D_PlasticBSDF
{
	float eta;
	float alpha;

	inline __device__ void PrepareBSDFData(const float3& wi,  const D_Material& material)
	{
		alpha = clamp((1.2f - 0.2f * sqrtf(fabs(wi.z))) * material.dielectric.roughness * material.dielectric.roughness, 1.0e-4f, 1.0f);
		eta = wi.z < 0.0f ? material.dielectric.ior : 1 / material.dielectric.ior;
	}

	// Evaluation function for a shadow ray
	inline __device__ bool Eval(const D_HitResult& hitResult, const float3& wi, const float3& wo, float3& throughput, float& pdf)
	{
		const float wiDotN = wi.z;
		const float woDotN = wo.z;

		const bool reflected = wiDotN * woDotN > 0.0f;

		if (wiDotN * woDotN < 0.0f)
			return false;

		const float3 m = normalize(wo + wi);
		float cosThetaT;
		const float wiDotM = dot(wi, m);
		const float woDotM = dot(wo, m);
		const float F = Fresnel::DieletricReflectance(1.0f / hitResult.material.dielectric.ior, wiDotM, cosThetaT);
		const float G = Microfacet::Smith_G2(alpha, woDotN, wiDotN);
		const float D = Microfacet::BeckmannD(alpha, m.z);

		// BRDF times woDotN
		const float3 brdf = make_float3(F * G * D / (4.0f * fabs(wiDotN)));

		// Diffuse bounce
		const float3 btdf = (1.0f - F) * hitResult.material.plastic.albedo * INV_PI * wo.z;

		throughput = brdf + btdf;

		// pm * jacobian = pm * || dWhr / dWo ||
		const float pdfSpecular = D * m.z / (4.0f * fabs(wiDotM));

		// cos(theta) / PI
		const float pdfDiffuse = wo.z * INV_PI;

		pdf = F * pdfSpecular + (1.0f - F) * pdfDiffuse;
		
		if (pdf > 1.0e5f)
			return false;

		return true;
	}

	inline __device__ bool Sample(const D_HitResult& hitResult, const float3& wi, float3& wo, float3& throughput, float& pdf, unsigned int& rngState)
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

			const float weight = Microfacet::WeightBeckmannWalter(alpha, fabs(wiDotM), fabs(wo.z), fabs(wi.z), m.z);

			// Handle divisions by zero
			if (weight > 1.0e10)
				return false;

			// If the new ray is under the hemisphere, return
			if (wo.z * wi.z < 0.0f)
				return false;

			// We dont need to include the Fresnel term since it's already included when
			// we select between reflection and transmission (see paper page 7)
			throughput = make_float3(weight); // * F / fr
			pdf = Microfacet::SampleWalterReflectionPdf(alpha, m.z, fabs(wiDotM));
		}

		else
		{
			//Diffuse
			wo = Random::RandomCosineHemisphere(rngState);
			throughput = hitResult.material.dielectric.albedo;
			// Same here, we don't need to include the Fresnel term
			//throughput = throughput * (1.0f - F) / (1.0f - fr)
			pdf = INV_PI * wo.z;
		}
		return true;
	}
};
