#pragma once
#include "Memory/Host/Allocators/Allocator.h"
#include <cuda_runtime_api.h>
#include "Utils/Utils.h"

class DeviceAllocator: Allocator
{
public:
	DeviceAllocator() = default;

protected:
	virtual void* Alloc(size_t size) override
	{
		void* devicePtr;
		checkCudaErrors(cudaMalloc(&devicePtr, size));
		return devicePtr;
	}

	virtual void Free(void* ptr) override
	{
		checkCudaErrors(cudaFree(ptr));
	}
};