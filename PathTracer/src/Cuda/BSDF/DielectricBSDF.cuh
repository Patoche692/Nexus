#pragma once
#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Geometry/Material.h"
#include "Cuda/Random.cuh"
#include "Microfacet.cuh"
#include "Fresnel.cuh"

/* 
	Rough dielectric BSDF based on the paper "Microfacet Models for Refraction through Rough Surfaces"
	See https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
*/
struct DielectricBSDF 
{
	float F0;
	float eta;

	float alpha;
	float reflectance;

	inline __device__ void PrepareBSDFData(float3& wi,  Material& material)
	{
		//alpha = max(material.roughness * material.roughness, 1.0e-3);
		alpha = max(1.0e-3, (1.2f - 0.2f * sqrt(abs(wi.z))) * material.roughness);
		reflectance = material.iorLevel;

		// TODO: include etaI in ray structure
		float etaI = 1.0f;
		eta = wi.z < 0.0f ? material.ior / etaI : etaI / material.ior;
		F0 = (material.ior - etaI) / (etaI + material.ior);
		F0 *= F0;
		//F0 *= reflectance * 2;
	}

	inline __device__ bool Eval(HitResult& hitResult, float3& wi, float3& wo, float3& throughput, unsigned int& rngState, float3 gnormal)
	{
		float3 m;

		if (alpha == 0.0f)
		{
			// Perfect mirror
			m = make_float3(0.0f, 0.0f, 1.0f);
		}
		else
			m = Microfacet::SampleSpecularHalfBeckWalt(alpha, rngState);

		const float wiDotM = dot(wi, m);

		float cosThetaT;
		const float fr = Fresnel::DieletricReflectance(1 / hitResult.material.ior, wiDotM, cosThetaT);
		bool sampleT = hitResult.material.transmittance > 0.0f;

		float3 F = make_float3(fr);

		// Randomly select a reflected or diffuse ray based on Fresnel reflectance
		if (Random::Rand(rngState) < fr)
		{
			// Specular
			wo = reflect(-wi, m);

			// If the new ray is under the hemisphere, return
			if (wo.z * wi.z < 0.0f)
				return false;

			const float wiDotM = dot(wi, m);
			const float weight = Microfacet::WeightBeckmannWalter(alpha, abs(wiDotM), abs(wo.z), abs(wi.z), m.z);

			// Handle divisions by zero
			if (weight > 1.0e10)
				return false;

			throughput = F * weight / fr;
		}

		else
		{
			if (Random::Rand(rngState) < hitResult.material.transmittance)
			{
				// Transmission
				wo = (eta * wiDotM - Utils::SgnE(wiDotM) * cosThetaT) * m - eta * wi;

				if (wo.z * wi.z > 0.0f)
					return false;

				const float weight = Microfacet::WeightBeckmannWalter(alpha, abs(wiDotM), abs(wo.z), abs(wi.z), m.z);

				if (weight > 1.0e10)
					return false;

				throughput = hitResult.material.diffuse * weight;
			}
			else
			{
				//Diffuse
				wo = Random::RandomCosineHemisphere(rngState);
				throughput = hitResult.material.diffuse;
			}
			throughput = throughput * (1.0f - F) / (1.0f - fr);
		}

		return true;
	}
};