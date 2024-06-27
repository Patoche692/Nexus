#pragma once
#include "Allocators/DeviceAllocator.h"
#include "CudaMemory.h"
#include "DeviceInstance.h"

/*
 * Device vector for trivial types (does not handle copy / move constructors
 * since it is not possible to construct an object on the GPU from the host with CUDA).
 * The objects will just be copied over to the device, non trivial conversions
 * from a CPU to GPU instance must therefore be handled by the user.
 * 
 * If the specified type implements a static T* ToDevice() method
 */
template<typename THost, typename TDevice>
class DeviceVector
{
public:
	DeviceVector()
	{
		Realloc(2);
	}

	DeviceVector(size_t size, DeviceAllocator<TDevice>* allocator = nullptr)
		:m_Allocator(allocator)
	{
		Realloc(size);
	}

	~DeviceVector()
	{
		Clear();
		DeviceAllocator<TDevice>::Free(m_Allocator, m_Data);
	}

	void PushBack(const THost& value)
	{
		if (m_Size >= m_Capacity)
			Realloc(m_Capacity + m_Capacity / 2);

		if (is_trivially_copyable_to_device<THost>::value)
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
		if (!is_trivially_destructible_from_device<THost>::value)
			THost::FreeFromDevice(m_Data + m_Size);
	}

	void Clear()
	{
		if (!is_trivially_destructible_from_device<THost>::value)
		{
			for (size_t i = 0; i < m_Size; i++)
				THost::FreeFromDevice(m_Data + i);
		}

		m_Size = 0;
	}

	size_t Size() const { return m_Size; }

	TDevice* Data() const { return m_Data; }

	DeviceInstance<THost, TDevice> operator[] (size_t index)
	{
		assert(index > 0 && index < m_Size);
		return DeviceInstance<THost, TDevice>(m_Data + index);
	}

private:
	Realloc(size_t newCapacity)
	{
		TDevice* newBlock = DeviceAllocator<TDevice>::Alloc(m_Allocator, newCapacity);

		size_t size = std::min(newCapacity, m_Size);

		CudaMemory::Copy<TDevice>(newBlock, m_Data, size, cudaMemcpyDeviceToDevice);

		if (!is_trivially_destructible_from_device<THost>::value)
		{
			for (size_t i = 0; i < size; i++)
				THost::FreeFromDevice(m_Data + i);
		}

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

void f()
{
	DeviceVector<int, int> deviceVector(6);
	deviceVector[2] = 1;
}
