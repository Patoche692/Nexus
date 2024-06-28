#pragma once
#include "Allocators/DeviceAllocator.h"
#include "CudaMemory.h"
#include "DeviceInstance.h"
#include "Memory/Host/Vector.h"

/*
 * Device vector for trivial types (does not handle copy / move constructors
 * since it is not possible to construct an object on the GPU from the host with CUDA).
 * The objects will just be copied over to the device, non trivial conversions
 * from a CPU to GPU instance must therefore be handled by the user.
 * 
 * If THost implements a static TDevice ToDevice() method, this method 
 * will be used to create a device instance and copy it to the device
 * when using the assignment operator.
 * Likewise, if THost implements a static void DestructFromDevice() method,
 * it will be used for destructing the device instance.
 */
template<typename THost, typename TDevice = THost>
class DeviceVector
{
public:
	DeviceVector()
	{
		Realloc(2);
	}

	DeviceVector(size_t size, DeviceAllocator<TDevice>* allocator = nullptr)
		:m_Allocator(allocator), m_Size(size)
	{
		Realloc(size);
	}

	//DeviceVector(const Vector<THost>& hostVector, DeviceAllocator<TDevice>* allocator = nullptr)
	//	:m_Allocator(allocator), m_Size(hostVector.Size())
	//{
	//	Realloc(size);
	//}

	~DeviceVector()
	{
		Clear();
		DeviceAllocator<TDevice>::Free(m_Allocator, m_Data);
	}

	void PushBack(const THost& value)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		if constexpr (is_trivially_copyable_to_device<THost>)
			CudaMemory::Copy<TDevice>(m_Data + m_Size, (TDevice*)&value, 1, cudaMemcpyHostToDevice);
		else
		{
			TDevice deviceInstance = THost::ToDevice(value);
			CudaMemory::Copy<TDevice>(m_Data + m_Size, &deviceInstance, 1, cudaMemcpyHostToDevice);
		}
		m_Size++;
	}

	void PopBack()
	{
		assert(m_Size > 0);
		m_Size--;
		if constexpr (!is_trivially_destructible_from_device<THost>)
			THost::DestructFromDevice(DeviceInstance<THost, TDevice>(m_Data + m_Size).Get());
	}

	void Clear()
	{
		if constexpr (!is_trivially_destructible_from_device<THost>)
		{
			for (size_t i = 0; i < m_Size; i++)
				THost::DestructFromDevice(DeviceInstance<THost, TDevice>(m_Data + i).Get());
		}

		m_Size = 0;
	}

	size_t Size() const { return m_Size; }

	TDevice* Data() const { return m_Data; }

	DeviceInstance<THost, TDevice> operator[] (size_t index)
	{
		assert(index >= 0 && index < m_Size);
		return DeviceInstance<THost, TDevice>(m_Data + index);
	}

private:
	void Realloc(size_t newCapacity)
	{
		TDevice* newBlock = DeviceAllocator<TDevice>::Alloc(m_Allocator, newCapacity);

		size_t size = std::min(newCapacity, m_Size);

		CudaMemory::Copy<TDevice>(newBlock, m_Data, size, cudaMemcpyDeviceToDevice);

		DeviceAllocator<TDevice>::Free(m_Allocator, m_Data);
		m_Data = newBlock;
		m_Capacity = newCapacity;
	}

private:
	TDevice* m_Data = nullptr;
	DeviceAllocator<TDevice>* m_Allocator = nullptr;

	size_t m_Size = 0;
	size_t m_Capacity = 0;
};

