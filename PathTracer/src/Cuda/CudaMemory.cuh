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
	static void MemCpy(T* dst, T* src, uint32_t count, cudaMemcpyKind kind)
	{
		checkCudaErrors(cudaMemcpy(dst, src, sizeof(T) * count, kind));
	}

	static void Free(void* ptr)
	{
		checkCudaErrors(cudaFree(ptr));
	}

	template<typename T>
	static void ResizeDeviceArray(T** symbolAddress, uint32_t size)
	{
		T* deviceArray;

		// Retrieve the address pointed to by the symbol
		MemCpy(&deviceArray, symbolAddress, 1, cudaMemcpyDeviceToHost);

		T* newArray = Allocate<T>(size);

		MemCpy(newArray, deviceArray, size - 1, cudaMemcpyHostToHost);
		MemCpy(symbolAddress, &newArray, 1, cudaMemcpyHostToDevice);

		if (size > 1)
			Free(deviceArray);

	}

	template<typename T>
	static void SetToIndex(T** symbolAddress, uint32_t index, T& element)
	{
		T* deviceArray;
		// Retrieve the address pointed to by the symbol
		MemCpy(&deviceArray, symbolAddress, 1, cudaMemcpyDeviceToHost);

		MemCpy(deviceArray + index, &element, 1, cudaMemcpyHostToDevice);
	}
};