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
	enum { value = sizeof(test<T>(0)) != sizeof(char) };
};

template<typename T>
class is_trivially_destructible_from_device
{
	typedef char one;
	struct two { char x[2]; };

	template<typename C> static one test(decltype(&C::FreeFromDevice));
	template<typename C> static two test(...);

public:
	enum { value = sizeof(test<T>(0)) != sizeof(char) };
};

/*
 * Wrapper class holding a device pointer to the device instance.
 * THost is the host class, TDevice is the device class, both can be identical.
 * If THost implements a static TDevice* ToDevice() method, this method 
 * will be used to create a device instance and copy it to the device
 * when using the assignment operator.
 * Likewise, if THost implements a static void FreeFromDevice() method,
 * it will be used before deallocating the device instance
 */
template<typename THost, typename TDevice = THost>
class DeviceInstance
{
public:
	DeviceInstance() = delete;

	DeviceInstance(TDevice* devicePtr)
		: m_DevicePtr(devicePtr), m_OwnsPtr(false) { }

	DeviceInstance(const THost& hostInstance)
		: m_OwnsPtr(true)
	{
		m_DevicePtr = CudaMemory::Allocate<THost>(1);
		SetDeviceInstance(hostInstance);
	}

	DeviceInstance(const DeviceInstance<THost, TDevice>& other)
	{
		assert(is_trivially_copyable_to_device<THost>::value);
		m_OwnsPtr = true;
		m_DevicePtr = CudaMemory::Allocate<THost>(1);
		CudaMemory::Copy<TDevice>(other.m_DevicePtr, m_DevicePtr, 1, cudaMemcpyDeviceToDevice);
	}

	DeviceInstance(DeviceInstance<THost, TDevice>&& other)
	{
		m_OwnsPtr = other.m_OwnsPtr;
		m_DevicePtr = other.m_DevicePtr;
		other.m_DevicePtr = nullptr;
	}

	~DeviceInstance()
	{
		if (m_OwnsPtr && m_DevicePtr)
		{
			CudaMemory::Free(m_DevicePtr);
			m_DevicePtr = nullptr;
		}
	}

	void operator=(const THost& hostInstance)
	{
		SetDeviceInstance(hostInstance);
	}

	TDevice Get()
	{
		TDevice target;
		CudaMemory::Copy(&target, m_DevicePtr, 1, cudaMemcpyDeviceToHost);
		return target;
	}

private:
	void SetDeviceInstance(const THost& hostInstance)
	{
		if (!is_trivially_destructible_from_device<THost>::value)
			THost::FreeFromDevice(Get());

		if (!is_trivially_copyable_to_device<THost>::value)
		{
			TDevice deviceInstance = THost::ToDevice(hostInstance);
			CudaMemory::Copy<TDevice>(m_DevicePtr, &deviceInstance, cudaMemcpyHostToDevice);
		}
		else
			CudaMemory::Copy<TDevice>(m_DevicePtr, (TDevice*)&hostInstance, 1, cudaMemcpyHostToDevice);
	}

private:
	TDevice* m_DevicePtr = nullptr;
	bool m_OwnsPtr = false;
};
