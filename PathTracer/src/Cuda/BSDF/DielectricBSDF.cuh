#pragma once
#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"
#include "Geometry/Material.h"
#include "Cuda/Random.cuh"
#include "Microfacet.cuh"
#include "Fresnel.cuh"

struct DielectricBSDF 
{
	float F0;
	float eta;

	float alpha;
	float reflectance;

	inline __device__ void PrepareBSDFData(float3& Vlocal,  Material& material)
	{
		alpha = material.roughness * material.roughness;
		reflectance = material.iorLevel;

		// TODO: include etaI in ray structure
		float etaI = 1.0f;
		eta = Vlocal.z < 0 ? material.ior / etaI : etaI / material.ior;
		F0 = (material.ior - etaI) / (etaI + material.ior);
		F0 *= F0;
		F0 *= reflectance * 2;
	}

	inline __device__ bool Eval(HitResult& hitResult, float3& wi, float3& wo, float3& throughput, unsigned int& rngState)
	{
		float3 m;

		if (alpha == 0.0f)
		{
			// Perfect mirror
			m = make_float3(0.0f, 0.0f, 1.0f);
		}
		else
			m = Microfacet::SampleSpecularHalfBeckWalt(alpha, rngState);

		float wiDotM = dot(wi, m);

		float fr = Fresnel::SchlickDielectricReflectance(F0, Fresnel::ShadowedF90(make_float3(F0)), wiDotM);
		float3 F = make_float3(fr);

		// Randomly select a reflected or diffuse ray based on Fresnel reflectance
		if (Random::Rand(rngState) < fr)
		{
			// Specular
			wo = reflect(-wi, m);

			// If the new ray is under the hemisphere, return
			if (wo.z <= 0.0f)
				return false;

			float wiDotM = dot(wi, m);
			float weight = Microfacet::WeightBeckmannWalter(alpha, wiDotM, wo.z, wi.z, m.z);

			// Handle divisions by zero
			if (weight > 1e10)
				return false;

			throughput = F * weight / fr;
		}

		else
		{
			if (Random::Rand(rngState) < hitResult.material.transmittance)
			{
				// Transmission
				wo = reflect(wi, m);
				wo.z *= -1;
				float woDotM = abs(dot(wo, m));
				float weight = Microfacet::WeightBeckmannWalter(alpha, woDotM, wo.z, wi.z, m.z);

				if (weight > 1e10)
					return false;

				throughput = hitResult.material.diffuse * weight;

				// If the new ray is in the upper hemisphere, return
				if (wo.z >= 0.0f)
					return false;
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