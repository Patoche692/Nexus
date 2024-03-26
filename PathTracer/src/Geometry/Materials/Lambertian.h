#pragma once
#include "Material.h"
#include "../../Cuda/Random.cuh"

//struct Lambertian : public Material
//{
//public:
//	__host__ __device__ Lambertian() = default;
//	__host__ __device__ Lambertian(float3 c) : albedo(c) {}
//
//	inline __host__ __device__ bool Scatter(float3& p, float3& attenuation, float3& normal, Ray& scattered, uint32_t& rngState) override {
//		float3 scatterDirection = normal + Random::RandomUnitVector(rngState);
//		scattered = Ray(p + normal * 0.001f, scatterDirection);
//		attenuation *= albedo;
//		return true;
//	}
//
//	virtual __host__ __device__ size_t GetSize() override { return sizeof(*this); }
//
//	float3 albedo;
//};
