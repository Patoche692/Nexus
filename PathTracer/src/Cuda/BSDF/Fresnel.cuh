#include <cuda_runtime_api.h>
#include "Utils/cuda_math.h"

class Fresnel
{
public:
	// Fresnel reflectance for dieletric materials. See https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
	inline static __device__ float DieletricReflectance(const float eta, const float cosThetaI, float& cosThetaT)
	{
		float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

		if (sinThetaTSq > 1.0f)
		{
			cosThetaT = 0.0f;
			return 1.0f;
		}

		cosThetaT = sqrt(fmax(0.0f, 1.0f - sinThetaTSq));

		float Rparl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
		float Rperp = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaT);

		return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
	}

	// Schlick reflectance approximation
	inline static __device__ float SchlickDielectricReflectance(const float R0, const float R90, const float cosThetaI)
	{
		return R0 + (R90 - R0) * pow(1.0f - cosThetaI, 5.0f);
	}

	// Custom fresnel reflectance at grazing angles (90°) depending on R0. See https://boksajak.github.io/files/CrashCourseBRDF.pdf
	inline static __device__ float ShadowedF90(const float3 F0) {
		const float t = (1.0f / 0.04f);
		return min(1.0f, t * Luminance(F0));
	}

private:
	inline static __device__ float Luminance(const float3 rgb)
	{
		return dot(rgb, make_float3(0.2126f, 0.7152f, 0.0722f));
	}

};