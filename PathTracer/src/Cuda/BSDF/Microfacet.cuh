#include <cuda_runtime_api.h>
#include "Utils/Utils.h"
#include "Utils/cuda_math.h"
#include "CUDA/Random.cuh"

class Microfacet 
{
public:

	inline static __device__ float BeckmannD(const float alpha, const float mDotN)
	{
		float alphaSq = alpha * alpha;
		float cosThetaSq = mDotN * mDotN;
		float numerator = exp((cosThetaSq - 1.0f) / (alphaSq * cosThetaSq));
		float denominator = M_PI * alphaSq * cosThetaSq * cosThetaSq;
		return numerator / denominator;
	}

	inline static __device__ float Smith_G_a(const float alpha, const float sDotN) {
		return sDotN / (max(0.00001f, alpha) * sqrt(1.0f - min(0.99999f, sDotN * sDotN)));
	}

	inline static __device__ float Smith_G1_Beckmann_Walter(const float a) {
		if (a < 1.6f) {
			return ((3.535f + 2.181f * a) * a) / (1.0f + (2.276f + 2.577f * a) * a);
		}
		else {
			return 1.0f;
		}
	}

	inline static __device__ float Smith_G1_Beckmann_Walter(const float alpha, const float sDotN, const float alphaSquared, const float NdotSSquared)
	{
		return Smith_G1_Beckmann_Walter(Smith_G_a(alpha, sDotN));
	}

	inline static __device__ float Smith_G2(const float alpha, const float NdotL, const float wiDotN)
	{
		float aL = Smith_G_a(alpha, NdotL);
		float aV = Smith_G_a(alpha, wiDotN);
		return Smith_G1_Beckmann_Walter(aL) * Smith_G1_Beckmann_Walter(aV);
	}

	// Weight of the sample for a Walter Beckmann sampling
	inline static __device__ float WeightBeckmannWalter(
		const float alpha, const float wiDotM, const float woDotN,
		const float wiDotN, const float mDotN
	) {
		return (wiDotM * Smith_G2(alpha, woDotN, wiDotN)) / (wiDotN * mDotN);
	}

	inline static __device__ float SampleWalterReflectionPdf(const float alpha, const float mDotN, const float woDotM)
	{
		return BeckmannD(max(0.00001f, alpha), mDotN) * mDotN / (4.0f * woDotM);
	}

	inline static __device__ float3 SampleSpecularHalfBeckWalt(const float alpha, unsigned int& rngState)
	{
		float a = dot(make_float2(alpha), make_float2(0.5f, 0.5f));

		float2 u = make_float2(Random::Rand(rngState), Random::Rand(rngState));
		float tanThetaSquared = -(a * a) * log(1.0f - u.x);
		float phi = TWO_TIMES_PI * u.y;

		// Calculate cosTheta and sinTheta needed for conversion to H vector
		float cosTheta = 1.0 / sqrt(1.0f + tanThetaSquared);
		float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

		// Convert sampled spherical coordinates to H vector
		return normalize(make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta));
	}
};