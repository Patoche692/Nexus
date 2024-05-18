#pragma once

#include <cuda_runtime_api.h>
#include <iostream>
#include "cuda_math.h"

#define M_PI  3.14159265358979323846
#define INV_PI 0.31830988618f
#define TWO_TIMES_PI 6.28318530718f

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace Utils
{
	template<typename T>
	inline __host__ __device__ void Swap(T& a, T& b) 
	{
		T c = a;
		a = b;
		b = c;
	}

	template<typename T>
	inline __host__ __device__ T SgnE(T val)
	{
		return val < T(0) ? T(-1) : T(1);
	}

	inline __host__ __device__ float ToRadians(float angle)
	{
		return angle * M_PI / 180.0f;
	}

	inline __host__ __device__ float ToDegrees(float angle)
	{
		return angle * 180.0f / M_PI;
	}

	inline __host__ __device__ float3 LinearToGamma(const float3& color)
	{
		return make_float3(pow(color.x, 0.45454545454), pow(color.y, 0.45454545454), pow(color.z, 0.45454545454));
	}

	inline __host__ __device__ float3 GammaToLinear(const float3& color)
	{
		return make_float3(pow(color.x, 2.2), pow(color.y, 2.2), pow(color.z, 2.2));
	}

	void GetPathAndFileName(const std::string fullPath, std::string& path, std::string& name);
}
