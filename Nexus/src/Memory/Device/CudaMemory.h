#pragma once

#include "Utils/Utils.h"
#include <cuda_runtime_api.h>


class CudaMemory
{
public:
	template<typename T>
	static T* Allocate(uint32_t count)
	{
		T* ptr;
		checkCudaErrors(cudaMalloc(&ptr, sizeof(T) * count));
		return ptr;
	}

	template<typename T>
	static void Copy(T* dst, T* src, uint32_t count, cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * count, kind));
	}

	template<typename T>
	static void Free(T* ptr)
	{
		checkCudaErrors(cudaFree(ptr));
	}
};