#pragma once

#include "../Utils/Utils.h"
#include <cuda_runtime_api.h>

class MemoryHelper
{
public:
	template<typename T>
	static void ResizeDeviceArray(T** symbolAddress, uint32_t size)
	{
		T* deviceArray;
		T* hostCpyArray = new T[size];

		// Retrieve the address pointed to by the symbol
		checkCudaErrors(cudaMemcpy(&deviceArray, symbolAddress, sizeof(T*), cudaMemcpyDeviceToHost));

		// Copy the array to hostCpyArray
		checkCudaErrors(cudaMemcpy(hostCpyArray, deviceArray, sizeof(T) * (size - 1), cudaMemcpyDeviceToHost));

		if (size > 1)
			checkCudaErrors(cudaFree(deviceArray));

		T* temp;
		checkCudaErrors(cudaMalloc((void**)&temp, sizeof(T) * size));

		checkCudaErrors(cudaMemcpy(temp, hostCpyArray, sizeof(T) * size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(symbolAddress, &temp, sizeof(T*), cudaMemcpyHostToDevice));

		delete[] hostCpyArray;
	}

	template<typename T>
	static void SetToIndex(T** symbolAddress, uint32_t index, T& element)
	{
		T* deviceArray;
		// Retrieve the address pointed to by the symbol
		checkCudaErrors(cudaMemcpy(&deviceArray, symbolAddress, sizeof(T*), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(deviceArray + index, &element, sizeof(T), cudaMemcpyHostToDevice));
	}

};