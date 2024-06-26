#pragma once
#include <cuda_runtime_api.h>
#include "Memory/Device/CudaMemory.h"


/*
 * Checks for an existing implementation of a ToDevice() static method using SFINAE.
 * See https://stackoverflow.com/questions/257288/how-can-you-check-whether-a-templated-class-has-a-member-function
 */
template<typename T>
class is_trivially_copyable_to_device
{
	typedef char one;
	struct two { char x[2]; };

	template<typename C> static one test(decltype(&C::ToDevice));
	template<typename C> static two test(...);

public:
	enum { value = sizeof(test<T>(0)) == sizeof(char) };
};


/*
 * Wrapper class holding a device pointer to the device instance
 */
template<typename T>
class DeviceInstance
{
public:
	DeviceInstance() = delete;

	DeviceInstance(T* devicePtr)
		: m_DevicePtr(devicePtr) { }

	void operator=(const T& hostInstance)
	{
		if (!is_trivially_copyable_to_device<T>::value)
			T::FreeFromDevice(m_DevicePtr);

		CudaMemory::Copy(m_DevicePtr, &hostInstance, 1, cudaMemcpyHostToDevice);
	}

	T Get()
	{
		T target;
		CudaMemory::Copy(&target, m_DevicePtr, 1, cudaMemcpyDeviceToHost);
		return target;
	}

private:
	T* m_DevicePtr = nullptr;
};
