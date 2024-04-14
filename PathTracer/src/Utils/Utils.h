#pragma once

#include <cuda_runtime_api.h>
#include <iostream>

#define M_PI  3.14159265358979323846
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

namespace Utils
{
	template<typename T>
	inline __host__ __device__ void Swap(T& a, T& b) {
		T c = a;
		a = b;
		b = c;
	}
}
