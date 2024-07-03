#pragma once
#include <cuda_runtime_api.h>
#include "Memory/Device/CudaMemory.h"


/*
 * Checks for an existing implementation of a ToDevice() method using SFINAE.
 * See https://stackoverflow.com/questions/257288/how-can-you-check-whether-a-templated-class-has-a-member-function
 */
template<typename T>
class ImplementsToDevice
{
	typedef char one;
	struct two { char x[2]; };

	template<typename C> static one test(decltype(&C::ToDevice));
	template<typename C> static two test(...);

public:
	enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<typename T>
class ImplementsDestructFromDevice
{
	typedef char one;
	struct two { char x[2]; };

	template<typename C> static one test(decltype(&C::DestructFromDevice));
	template<typename C> static two test(...);

public:
	enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template<typename T>
constexpr bool is_trivially_copyable_to_device = !ImplementsToDevice<T>::value;

template<typename T>
constexpr bool is_trivially_destructible_from_device = !ImplementsDestructFromDevice<T>::value;


/*
 * Wrapper class holding a device pointer to the device instance.
 * THost is the host class, TDevice is the device class, both can be identical.
 * 
 * If THost implements a static TDevice ToDevice() method, this method 
 * will be used to create a device instance and copy it to the device
 * when using the assignment operator.
 * Likewise, if THost implements a static void DestructFromDevice() method,
 * it will be used for destructing the device instance.
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
		m_DevicePtr = CudaMemory::Allocate<TDevice>(1);
		SetDeviceInstance(hostInstance);
	}

	DeviceInstance(const DeviceInstance<THost, TDevice>& other)
	{
		assert(is_trivially_copyable_to_device<THost>);
		m_OwnsPtr = true;
		m_DevicePtr = CudaMemory::Allocate<TDevice>(1);
		CudaMemory::Copy<TDevice>(other.m_DevicePtr, m_DevicePtr, 1, cudaMemcpyDeviceToDevice);
	}

	DeviceInstance(DeviceInstance<THost, TDevice>&& other)
		: m_OwnsPtr(other.m_OwnsPtr), m_DevicePtr(other.m_DevicePtr)
	{
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
		DestructDeviceInstance();
		SetDeviceInstance(hostInstance);
	}

	TDevice Get()
	{
		TDevice target;
		CudaMemory::Copy(&target, m_DevicePtr, 1, cudaMemcpyDeviceToHost);
		return target;
	}

	TDevice* Data() { return m_DevicePtr; }

private:

	void SetDeviceInstance(const THost& hostInstance)
	{
		if constexpr (!is_trivially_copyable_to_device<THost>)
		{
			TDevice deviceInstance = THost::ToDevice(hostInstance);
			CudaMemory::Copy<TDevice>(m_DevicePtr, &deviceInstance, 1, cudaMemcpyHostToDevice);
		}
		else
			CudaMemory::Copy<TDevice>(m_DevicePtr, (TDevice*)&hostInstance, 1, cudaMemcpyHostToDevice);
	}

	void DestructDeviceInstance()
	{
		if constexpr (!is_trivially_destructible_from_device<THost>)
			THost::DestructFromDevice(Get());
	}

private:
	TDevice* m_DevicePtr = nullptr;
	bool m_OwnsPtr = false;
};


