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
		checkCudaErrors(cudaMalloc((void**)&ptr, sizeof(T) * count));
		return ptr;
	}

	template<typename T>
	static void Copy(T* dst, T* src, uint32_t count, cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, sizeof(T) * count, kind));
	}

	static void Free(void* ptr)
	{
		checkCudaErrors(cudaFree(ptr));
	}
};